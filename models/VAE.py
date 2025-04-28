from VAE_utils import (
    find_best_vae_architecture,
    train_vae_model,
    load_and_transform_with_vae,
    evaluate_and_visualize_vae
)

# 1. Paths
original_data_path = "data/SimCLR/SimCLR_data.npy"                # your raw SimCLR embeddings + labels
best_json_path     = "data/VAE/best_vae_architecture.json"        # where Optuna saved hyperparams
vae_model_path     = "data/VAE/final_vae_pcdarts.pth"             # where train_vae_model saved .pth
vae_transformed    = "data/VAE/vae_transformed.npy"               # where train_vae_model saved latents+labels
output_dir         = "data/VAE/plots"                             # folder to dump the PNGs

# 2. Run Optuna search & save best architecture
find_best_vae_architecture(
    data_path=original_data_path,
    json_out=best_json_path,
    n_trials=2
)

# 3. Train final VAE & save model + transformed data
train_vae_model(
    json_path=best_json_path,
    data_path=original_data_path,
    model_out=vae_model_path,
    transformed_out=vae_transformed
)

# 4. (Optional) Re‐load & re‐transform new data
# Z_new, labels_new = load_and_transform_with_vae(best_json_path, vae_model_path, new_data_path)

# 5. Evaluate & save all visualizations
evaluate_and_visualize_vae(
    json_path=best_json_path,
    original_data_path=original_data_path,
    transformed_path=vae_transformed,
    model_path=vae_model_path,
    output_dir=output_dir
)
