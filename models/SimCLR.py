from SimCLR_utils import (
    search_best_architecture,
    train_final_model,
    transform_data,
    evaluate_and_plot,
)
import os

BASE = "data/SimCLR"

# 1) Search and save best architecture
best = search_best_architecture(
    train_npy_path    = os.path.join(BASE, "train_cls_embeddings_with_label.npy"),
    output_json_path  = os.path.join(BASE, "best_arch.json"),
    n_trials          = 5
)

# 2) Train final model
model = train_final_model(
    train_npy_path    = os.path.join(BASE, "train_cls_embeddings_with_label.npy"),
    arch_json_path    = os.path.join(BASE, "best_arch.json"),
    output_model_path = os.path.join(BASE, "simclr_head.pth"),
    epochs= 5
)

# 3) Transform data
sim_data = transform_data(
    model_path       = os.path.join(BASE, "simclr_head.pth"),
    input_npy_path   = os.path.join(BASE, "test_cls_embeddings_with_label.npy"),
    output_npy_path  = os.path.join(BASE, "SimCLR_data.npy"),
    arch_json_path   = os.path.join(BASE, "best_arch.json")      # <— added here
)

# 4) Evaluate & plot
evaluate_and_plot(
    original_npy = os.path.join(BASE, "test_cls_embeddings_with_label.npy"),
    simclr_npy   = os.path.join(BASE, "SimCLR_data.npy"),
    output_fig   = os.path.join(BASE, "visualizations", "comparison.png")
)
