import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import optuna
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# -----------------------------------------------------------------------------
# 0. Helper: load (N, 769) → feats (N,768), labels (N,)
# -----------------------------------------------------------------------------
def load_embeddings_with_labels(path: str):
    arr    = np.load(path)                  # shape (N, 769)
    feats  = arr[:, :-1].astype(np.float32) # first 768 dims
    labels = arr[:, -1].astype(np.int64)    # last column
    return feats, labels

# -----------------------------------------------------------------------------
# PC-DARTS Projection Head & SimCLR Loss
# -----------------------------------------------------------------------------
class MixedOp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh()),
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid()),
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True)),
            nn.Linear(in_dim, out_dim)
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x):
        weights = F.softmax(self.alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class ProjectionHeadPCDARTS(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, num_cells, num_mixed_ops):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_cells
        # Each cell is a list of MixedOp modules
        self.cells = nn.ModuleList([
            nn.ModuleList([MixedOp(dims[i], dims[i+1]) for _ in range(num_mixed_ops)])
            for i in range(num_cells)
        ])
        self.out_layer = nn.Linear(dims[-1], proj_dim)

    def forward(self, x):
        for cell in self.cells:
            x = sum(op(x) for op in cell) / len(cell)
        return self.out_layer(x)

def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim[~mask].view(2*B, -1)
    pos = torch.cat([torch.sum(z1*z2, dim=1), torch.sum(z1*z2, dim=1)], dim=0).unsqueeze(1)
    logits = torch.cat([pos, sim], dim=1) / temperature
    labels = torch.zeros(2*B, device=z.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)

# -----------------------------------------------------------------------------
# 1. SEARCH: PC-DARTS + Optuna to find best SimCLR architecture
# -----------------------------------------------------------------------------
def search_best_architecture(train_npy_path, output_json_path, n_trials=20):
    feats, _ = load_embeddings_with_labels(train_npy_path)

    def objective(trial):
        num_cells     = trial.suggest_int("num_cells", 2, 4)
        num_mixed_ops = trial.suggest_int("num_mixed_ops", 1, 4)
        hidden_dim    = trial.suggest_int("hidden_dim", 128, 512)
        proj_dim      = trial.suggest_int("proj_dim", 32, 128)
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        temp          = trial.suggest_float("temperature", 0.1, 1.0)
        noise_std     = trial.suggest_float("noise_std", 0.05, 0.2)
        batch_size    = trial.suggest_int("batch_size", 32, 128)

        idx = np.random.choice(len(feats), batch_size, replace=False)
        x = torch.tensor(feats[idx], dtype=torch.float32)
        model = ProjectionHeadPCDARTS(feats.shape[1], hidden_dim, proj_dim,
                                     num_cells, num_mixed_ops)
        z1 = model(x + torch.randn_like(x)*noise_std)
        z2 = model(x + torch.randn_like(x)*noise_std)
        return nt_xent_loss(z1, z2, temp).item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params

    with open(output_json_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best architecture saved to {output_json_path}")
    return best

# -----------------------------------------------------------------------------
# 2. TRAIN: load JSON config, train full SimCLR projection head
# -----------------------------------------------------------------------------
def train_final_model(train_npy_path, arch_json_path, output_model_path,
                      epochs=20):
    feats, _ = load_embeddings_with_labels(train_npy_path)
    with open(arch_json_path) as f:
        cfg = json.load(f)

    model = ProjectionHeadPCDARTS(feats.shape[1],
                                 cfg["hidden_dim"],
                                 cfg["proj_dim"],
                                 cfg["num_cells"],
                                 cfg["num_mixed_ops"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ds = TensorDataset(torch.tensor(feats, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model.train()
    for ep in range(epochs):
        total = 0
        for (batch,) in loader:
            z1 = model(batch + torch.randn_like(batch)*cfg["noise_std"])
            z2 = model(batch + torch.randn_like(batch)*cfg["noise_std"])
            loss = nt_xent_loss(z1, z2, cfg["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()*batch.size(0)
        print(f"Epoch {ep+1}/{epochs}, Loss: {total/len(ds):.4f}")

    torch.save(model.state_dict(), output_model_path)
    print(f"Trained model saved to {output_model_path}")
    return model

# -----------------------------------------------------------------------------
# 3. TRANSFORM: use saved model to encode data + save SimCLR_data.npy
# -----------------------------------------------------------------------------
def transform_data(model_path, input_npy_path, output_npy_path, arch_json_path):
    feats, labels = load_embeddings_with_labels(input_npy_path)
    with open(arch_json_path) as f:
        cfg = json.load(f)

    model = ProjectionHeadPCDARTS(feats.shape[1],
                                 cfg["hidden_dim"],
                                 cfg["proj_dim"],
                                 cfg["num_cells"],
                                 cfg["num_mixed_ops"])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        z = model(torch.tensor(feats, dtype=torch.float32))

    simclr_data = np.hstack([z.numpy(), labels.reshape(-1,1)])
    np.save(output_npy_path, simclr_data)
    print(f"SimCLR embeddings + labels saved to {output_npy_path}")
    return simclr_data

# -----------------------------------------------------------------------------
# 4. EVALUATE: plot t-SNE & UMAP before/after in one figure
# -----------------------------------------------------------------------------
def evaluate_and_plot(original_npy, simclr_npy, output_fig, sample_size=2000):
    feats0, labels = load_embeddings_with_labels(original_npy)
    feats1, _      = load_embeddings_with_labels(simclr_npy)

    idx = np.random.choice(len(labels), min(sample_size, len(labels)), replace=False)
    f0, f1, lab = feats0[idx], feats1[idx,:-1], labels[idx]

    tsne0 = TSNE(n_components=2, random_state=42).fit_transform(f0)
    tsne1 = TSNE(n_components=2, random_state=42).fit_transform(f1)
    umap0 = umap.UMAP(random_state=42).fit_transform(f0)
    umap1 = umap.UMAP(random_state=42).fit_transform(f1)

    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    axs[0,0].scatter(tsne0[:,0], tsne0[:,1], c=lab, s=5); axs[0,0].set_title("t-SNE Before")
    axs[0,1].scatter(tsne1[:,0], tsne1[:,1], c=lab, s=5); axs[0,1].set_title("t-SNE After")
    axs[1,0].scatter(umap0[:,0], umap0[:,1], c=lab, s=5); axs[1,0].set_title("UMAP Before")
    axs[1,1].scatter(umap1[:,0], umap1[:,1], c=lab, s=5); axs[1,1].set_title("UMAP After")

    plt.tight_layout()
    fig.savefig(output_fig)
    print(f"Comparison figure saved to {output_fig}")
    plt.show()
