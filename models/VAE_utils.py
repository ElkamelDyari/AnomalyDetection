import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# ---------------------------------------
# 0. PC-DARTS Basic Modules
# ---------------------------------------
class OpLinearReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True))
    def forward(self, x): return self.op(x)

class OpLinearTanh(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
    def forward(self, x): return self.op(x)

class OpLinearSigmoid(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
    def forward(self, x): return self.op(x)

class OpLinearLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(negative_slope, inplace=True))
    def forward(self, x): return self.op(x)

class OpIdentity(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.op = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x): return self.op(x)

class MixedOp(nn.Module):
    CANDIDATES = [OpLinearReLU, OpLinearTanh, OpLinearSigmoid, OpLinearLeakyReLU, OpIdentity]
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ops = nn.ModuleList([cls(in_dim, out_dim) for cls in self.CANDIDATES])
        self.alpha = nn.Parameter(torch.randn(len(self.ops)))
    def forward(self, x):
        weights = F.softmax(self.alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class Cell(nn.Module):
    def __init__(self, in_dim, out_dim, num_mixed_ops):
        super().__init__()
        self.mixed_ops = nn.ModuleList([MixedOp(in_dim, out_dim) for _ in range(num_mixed_ops)])
    def forward(self, x):
        return sum(op(x) for op in self.mixed_ops) / len(self.mixed_ops)

# ---------------------------------------
# 1. VAE with PC-DARTS Encoder & Decoder
# ---------------------------------------
class VAE_PCDARTS(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim,
                 num_cells_enc, num_mixed_ops_enc, num_cells_dec, num_mixed_ops_dec):
        super().__init__()
        # Encoder
        current_dim = input_dim
        self.encoder_cells = nn.ModuleList()
        for _ in range(num_cells_enc):
            self.encoder_cells.append(Cell(current_dim, hidden_dim_enc, num_mixed_ops_enc))
            current_dim = hidden_dim_enc
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim_dec)
        current_dim = hidden_dim_dec
        self.decoder_cells = nn.ModuleList()
        for _ in range(num_cells_dec):
            self.decoder_cells.append(Cell(current_dim, hidden_dim_dec, num_mixed_ops_dec))
            current_dim = hidden_dim_dec
        self.reconstruction = nn.Linear(current_dim, input_dim)

    def encode(self, x):
        for cell in self.encoder_cells:
            x = cell(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_decoder(z))
        for cell in self.decoder_cells:
            x = cell(x)
        return self.reconstruction(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# ---------------------------------------
# 2. VAE Loss and Train Loop
# ---------------------------------------
def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(embeddings, model, optimizer, epochs, batch_size=64, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    model.to(device).train()
    loss_hist, recon_hist, kl_hist = [], [], []
    for epoch in range(epochs):
        total_loss = total_recon = total_kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = vae_loss(batch, x_recon, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_recon += recon_l.item() * batch.size(0)
            total_kl += kl_l.item() * batch.size(0)
        n = len(ds)
        loss_hist.append(total_loss/n)
        recon_hist.append(total_recon/n)
        kl_hist.append(total_kl/n)
        print(f"Epoch {epoch+1}/{epochs} Loss={loss_hist[-1]:.4f} Recon={recon_hist[-1]:.4f} KL={kl_hist[-1]:.4f}")
    return model, loss_hist, recon_hist, kl_hist

# ---------------------------------------
# 3. Four Required Functions
# ---------------------------------------
# 3.1 Find best VAE architecture & hyperparams
#     Saves best params to JSON
def find_best_vae_architecture(data_path, json_out, n_trials=20):
    arr = np.load(data_path)
    X = arr[:, :-1]
    def objective(trial):
        num_cells_enc = trial.suggest_int("num_cells_enc", 1, 3)
        num_mixed_enc = trial.suggest_int("num_mixed_ops_enc", 1, 3)
        hid_enc = trial.suggest_int("hidden_dim_enc", 64, 256)
        num_cells_dec = trial.suggest_int("num_cells_dec", 1, 3)
        num_mixed_dec = trial.suggest_int("num_mixed_ops_dec", 1, 3)
        hid_dec = trial.suggest_int("hidden_dim_dec", 64, 256)
        latent_dim = trial.suggest_int("latent_dim", 8, 64)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        beta = trial.suggest_float("beta", 0.5, 2.0)
        batch_size = trial.suggest_int("batch_size", 32, 128)
        epochs = 5
        model = VAE_PCDARTS(X.shape[1], hid_enc, hid_dec, latent_dim,
                            num_cells_enc, num_mixed_enc, num_cells_dec, num_mixed_dec)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        _, loss_hist, _, _ = train_vae(X, model, optimizer, epochs, batch_size, beta)
        return loss_hist[-1]
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params
    with open(json_out, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"Saved best VAE architecture to {json_out}")
    return best

# 3.2 Train VAE using best architecture JSON
def train_vae_model(json_path, data_path, model_out, transformed_out):
    with open(json_path, 'r') as f:
        params = json.load(f)
    arr = np.load(data_path)
    X, labels = arr[:, :-1], arr[:, -1]
    model = VAE_PCDARTS(
        input_dim=X.shape[1],
        hidden_dim_enc=params['hidden_dim_enc'],
        hidden_dim_dec=params['hidden_dim_dec'],
        latent_dim=params['latent_dim'],
        num_cells_enc=params['num_cells_enc'],
        num_mixed_ops_enc=params['num_mixed_ops_enc'],
        num_cells_dec=params['num_cells_dec'],
        num_mixed_ops_dec=params['num_mixed_ops_dec']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    model, loss_hist, recon_hist, kl_hist = train_vae(
        X, model, optimizer,
        epochs=5,
        batch_size=params['batch_size'],
        beta=params['beta']
    )
    torch.save(model.state_dict(), model_out)
    print(f"Saved VAE model weights to {model_out}")
    # transform data
    model.eval()
    # Move inputs to same device as model
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        mu, _ = model.encode(X_tensor)
        Z = mu.cpu().numpy()
    out = np.hstack([Z, labels.reshape(-1,1)])
    np.save(transformed_out, out)
    print(f"Saved transformed data to {transformed_out}")

# 3.3 Load saved VAE model and transform data
def load_and_transform_with_vae(json_path, model_path, data_path):
    # load architecture
    with open(json_path, 'r') as f:
        params = json.load(f)
    arr = np.load(data_path)
    X, labels = arr[:, :-1], arr[:, -1]
    model = VAE_PCDARTS(
        input_dim=X.shape[1],
        hidden_dim_enc=params['hidden_dim_enc'],
        hidden_dim_dec=params['hidden_dim_dec'],
        latent_dim=params['latent_dim'],
        num_cells_enc=params['num_cells_enc'],
        num_mixed_ops_enc=params['num_mixed_ops_enc'],
        num_cells_dec=params['num_cells_dec'],
        num_mixed_ops_dec=params['num_mixed_ops_dec']
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(X, dtype=torch.float32))
        Z = mu.cpu().numpy()
    return Z, labels

# 3.4 Evaluation & Visualization (saves figures)
def evaluate_and_visualize_vae(json_path, original_data_path, transformed_path, model_path, output_dir=None):
    """
    Reads JSON hyperparams, original and transformed data, and a saved VAE model.
    Saves t-SNE/UMAP comparison and reconstruction error histogram into output_dir.
    """
    # Determine output folder
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load original and transformed datasets
    arr_orig = np.load(original_data_path)
    X_orig, labels = arr_orig[:, :-1], arr_orig[:, -1]
    arr_trans = np.load(transformed_path)
    Z, _ = arr_trans[:, :-1], arr_trans[:, -1]

    # 1) t-SNE & UMAP before and after
    tsne_o = TSNE(n_components=2, random_state=42).fit_transform(X_orig)
    umap_o = umap.UMAP(random_state=42).fit_transform(X_orig)
    tsne_z = TSNE(n_components=2, random_state=42).fit_transform(Z)
    umap_z = umap.UMAP(random_state=42).fit_transform(Z)

    # 2) Plot comparison grid
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs[0,0].scatter(tsne_o[:,0], tsne_o[:,1], c=labels, s=5)
    axs[0,0].set_title('t-SNE Original')
    axs[0,1].scatter(tsne_z[:,0], tsne_z[:,1], c=labels, s=5)
    axs[0,1].set_title('t-SNE VAE')
    axs[1,0].scatter(umap_o[:,0], umap_o[:,1], c=labels, s=5)
    axs[1,0].set_title('UMAP Original')
    axs[1,1].scatter(umap_z[:,0], umap_z[:,1], c=labels, s=5)
    axs[1,1].set_title('UMAP VAE')
    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'vae_embeddings_comparison.png')
    fig.savefig(comp_path)
    plt.close(fig)
    print(f"Saved embeddings comparison to {comp_path}")

    # 3) Load model and compute reconstruction errors on original data
    with open(json_path,'r') as f:
        params = json.load(f)
    model = VAE_PCDARTS(
        input_dim=X_orig.shape[1],
        hidden_dim_enc=params['hidden_dim_enc'],
        hidden_dim_dec=params['hidden_dim_dec'],
        latent_dim=params['latent_dim'],
        num_cells_enc=params['num_cells_enc'],
        num_mixed_ops_enc=params['num_mixed_ops_enc'],
        num_cells_dec=params['num_cells_dec'],
        num_mixed_ops_dec=params['num_mixed_ops_dec']
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_orig, dtype=torch.float32)
        x_recon, mu, logvar = model(X_t)
        errors = F.mse_loss(x_recon, X_t, reduction='none').mean(dim=1).cpu().numpy()

    # 4) Plot reconstruction error histogram
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.hist(errors[labels==0], bins=50, alpha=0.5, label='Normal')
    ax2.hist(errors[labels==1], bins=50, alpha=0.5, label='Anomaly')
    ax2.set_title('Reconstruction Error Histogram')
    ax2.set_xlabel('Error'); ax2.set_ylabel('Count'); ax2.legend()
    hist_path = os.path.join(output_dir, 'vae_reconstruction_histogram.png')
    fig2.savefig(hist_path)
    plt.close(fig2)
    print(f"Saved reconstruction histogram to {hist_path}")