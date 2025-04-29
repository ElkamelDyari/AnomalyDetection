import pandas as pd
import numpy as np
import torch
# Note: Transformers models will be loaded lazily inside generate_cls_embeddings
try:
    # for type checking
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    pass
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os

# 2. Config
DATA_PATH   = "data/preprocessed/filtered_data.csv"
output_dir = "data/extracted_features"
MODEL_NAME  = "microsoft/codebert-base"
MAX_LEN     = 128
BATCH_SIZE  = 32
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# 5. Embedding generator
def generate_cls_embeddings(texts, batch_size=None):
    """
    Generate CodeBERT CLS token embeddings for a list of texts
    
    Args:
        texts: list of values (possibly non-str)
        batch_size: optional custom batch size to override global BATCH_SIZE
        
    Returns:
        np.array of shape (len(texts), 768)
    """
    # Lazy load CodeBERT tokenizer and model
    global tokenizer, model
    if 'tokenizer' not in globals() or 'model' not in globals():
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(device).eval()

    # Use custom batch size if provided, otherwise use global BATCH_SIZE
    actual_batch_size = batch_size or BATCH_SIZE
    
    # Free CUDA memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    all_embs = []
    for i in tqdm(range(0, len(texts), actual_batch_size), desc="Embedding"):
        raw_batch = texts[i : i + actual_batch_size]
        # FORCE every item to str:
        batch = [str(x) for x in raw_batch]

        # now safe to tokenize
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)

    return np.vstack(all_embs)

# 6. Main pipeline for CLI execution
def main():
    # 3. Load & subsample
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["template"]).astype(str)
    df["template"] = df["template"].astype(str)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # stratified split: 100K train, 150K temp
    train_df, temp_df = train_test_split(
        df,
        train_size=100_000,
        stratify=df["label"],
        random_state=SEED
    )
    # from temp, 100K test, 50K val
    test_df, val_df = train_test_split(
        temp_df,
        train_size=100_000,
        test_size=50_000,
        stratify=temp_df["label"],
        random_state=SEED
    )

    print("Sizes:", len(train_df), len(test_df), len(val_df))

    # Process splits & save
    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_df in [("train", train_df), ("test", test_df), ("val", val_df)]:
        print(f"--- Processing {split_name} split ({len(split_df)} rows) ---")
        texts = split_df["template"].tolist()
        labels = split_df["label"].to_numpy().reshape(-1, 1)
        embs = generate_cls_embeddings(texts)

        out = np.hstack([embs, labels])
        fname = f"{split_name}_cls_embeddings_with_label.npy"
        out_path = os.path.join(output_dir, fname)
        np.save(out_path, out)
        print(f"Saved {out_path} (shape {out.shape})")

if __name__ == "__main__":
    main()