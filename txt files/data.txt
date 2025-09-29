import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch

LABEL_COLUMNS = ['HateSpeech']

class CyberbullyingDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Ensure labels is 1D for single sample (shape: [1] for single label)
        labels = np.array(labels, dtype=np.float32)  # Ensure float32
        if labels.ndim == 0:  # Scalar case
            labels = np.array([labels])  # Convert to 1D array [label]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)  # Shape: [1]
        }

def load_data(dataset_path):
    """Load CSV with columns: Comments, HateSpeech. No preprocessing."""
    df = pd.read_csv(dataset_path, usecols=["Comments", "HateSpeech"])
    # If labels are "hate"/"nonhate", map them; otherwise just coerce to ints
    if df["HateSpeech"].dtype == object:
        mapping = {"hate": 1, "nonhate": 0, "1": 1, "0": 0}
        df["HateSpeech"] = df["HateSpeech"].map(mapping).astype("Int64")
    comments = df["Comments"].to_numpy()
    labels = df["HateSpeech"].to_numpy()  # shape (n,)
    return comments, labels

def prepare_kfold_splits(comments, labels, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    return kfold.split(comments)
