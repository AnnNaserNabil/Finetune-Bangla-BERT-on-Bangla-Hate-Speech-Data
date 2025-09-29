import torch
from transformers import BertTokenizer
import data
import model
import train
from config import parse_arguments
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def main():
    # Mount Google Drive only if in Colab
    if IN_COLAB:
        try:
            drive.mount('/content/drive', force_remount=True)  # Mount Google Drive with force_remount
            print("Google Drive mounted successfully.")
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            raise
    else:
        print("Not running in Colab; skipping Google Drive mount.")

    config = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(config.model_path)

    comments, labels = data.load_data(config.dataset_path)
    print(f"Loaded {len(comments)} samples with {len(data.LABEL_COLUMNS)} labels.")

    train.run_kfold_training(config, comments, labels, tokenizer, device)

if __name__ == "__main__":
    main()
