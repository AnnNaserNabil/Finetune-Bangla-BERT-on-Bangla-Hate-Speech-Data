# BanglaBERT Hate Speech Detection Fine-Tuning

![BanglaBERT](https://img.shields.io/badge/Model-BanglaBERT-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Project Overview

This project provides a modular Python framework for fine-tuning the [BanglaBERT model](https://huggingface.co/sagorsarker/bangla-bert-base) on a Bangla hate speech dataset for binary classification (`hate`/`nonhate`). It uses PyTorch, Hugging Face Transformers, and MLflow for experiment tracking. Key features include:

- **K-Fold Cross-Validation**: Default 5 folds for robust evaluation.
- **Checkpointing**: Saves model checkpoints after each epoch to Google Drive, deleting previous ones to save space, enabling resumption after Colab disconnections.
- **Class Weighting**: Handles imbalanced labels using `BCEWithLogitsLoss` with positive class weights.
- **Early Stopping**: Stops training if validation F1 score doesn’t improve (patience=5).
- **Freezing Base Layers**: Optional freezing of BERT layers for faster training and better generalization.
- **MLflow Tracking**: Logs hyperparameters, per-epoch metrics (accuracy, precision, recall, F1, loss), and models.

The framework is designed for reproducibility and collaboration, ideal for researchers experimenting with Bangla NLP tasks.

## Project Structure

```
Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/
├── config.py       # Command-line argument parsing
├── data.py         # Dataset loading and preprocessing
├── model.py        # BanglaBERT model definition
├── train.py        # Training and evaluation logic
├── main.py         # Entry point for running experiments
├── experiments.py   # Predefined hyperparameter configurations
├── data/
│   └── HateSpeech.csv  # Dataset (Comments, HateSpeech columns)
├── README.md       # This file
```

## Usage

### Running in Google Colab (Recommended)

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com).
2. **Enable GPU**: `Runtime > Change runtime type > Hardware accelerator: GPU (T4)`.
3. **Manually Mount Google Drive**:
   - Run in a Colab cell:
     ```bash
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Authenticate with your Google account, copy the authorization code, and paste it into the input box.
   - Verify mount:
     ```bash
     !ls /content/drive/MyDrive
     ```
4. **Clone Repository**:
   ```bash
   !git clone https://github.com/AnnNaserNabil/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data.git
   %cd Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data
   ```
5. **Install Dependencies**:
   ```bash
   !pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow google-colab
   ```
6. **Prepare Dataset**:
   - Ensure `HateSpeech.csv` is in `/content/drive/MyDrive/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data/`.
   - Copy to Colab:
     ```bash
     !mkdir -p /content/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data
     !cp /content/drive/MyDrive/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data/HateSpeech.csv /content/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data/
     ```
7. **Run Experiment**:
   - Choose a configuration from `experiments.py` (e.g., `{'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': True}`).
   - Run:
     ```bash
     !python main.py --batch 32 --lr 2e-5 --epochs 20 --author_name 'yourname' --dataset_path '/content/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data/HateSpeech.csv' --freeze_base --mlflow_experiment_name 'Bangla-HateSpeech-Experiments'
     ```
   - **Arguments**:
     - `--batch`: Batch size (e.g., 16, 32, 64).
     - `--lr`: Learning rate (e.g., 1e-5, 2e-5, 3e-5).
     - `--epochs`: Number of epochs (e.g., 10-30).
     - `--author_name`: Your name for MLflow run tagging.
     - `--dataset_path`: Path to `HateSpeech.csv`.
     - `--model_path`: Pre-trained model (default: `sagorsarker/bangla-bert-base`).
     - `--num_folds`: K-Fold splits (default: 5).
     - `--max_length`: Token length (default: 128).
     - `--freeze_base`: Freeze BERT base layers (optional).
     - `--mlflow_experiment_name`: MLflow experiment name (default: `Bangla-BERT-Cyberbullying`).
8. **Checkpoints**:
   - Saved to `/content/drive/MyDrive/checkpoints/checkpoint_fold_X.pt` after each epoch.
   - Previous epoch’s checkpoint is deleted to save space.
   - If Colab disconnects, re-run the command to resume from the last checkpoint.
9. **Download Results**:
   - Zip MLflow logs:
     ```bash
     !zip -r mlruns_yourname.zip ./mlruns
     ```
   - Download `mlruns_yourname.zip` from Colab’s file sidebar.

### Viewing Results Locally

1. Unzip `mlruns_yourname.zip` to a local directory (e.g., `experiments/mlruns_yourname`).
2. Navigate to the directory, activate a virtual environment, and run:
   ```bash
   mlflow ui
   ```
3. Open `http://localhost:5000` in a browser to view experiments, metrics (accuracy, precision, recall, F1, loss), parameters, and models.

### Running Locally (No Colab)

1. Clone the repo:
   ```bash
   git clone https://github.com/AnnNaserNabil/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data.git
   cd Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data
   ```
2. Install dependencies:
   ```bash
   pip install torch transformers scikit-learn pandas numpy tqdm mlflow
   ```
3. Ensure `HateSpeech.csv` is in the `data/` folder or specify its path.
4. Run:
   ```bash
   python main.py --batch 32 --lr 2e-5 --epochs 20 --author_name 'yourname' --dataset_path './data/HateSpeech.csv' --freeze_base --mlflow_experiment_name 'Bangla-HateSpeech-Experiments'
   ```

## Collaboration Guide

1. **Fork and Clone**: Fork the repo and clone to Colab or local machine.
2. **Experiment**: Open `experiments.py`, copy a config (e.g., `{'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': True}`), and run with your `--author_name` and `--dataset_path`.
3. **Add Configurations**: Add new hyperparameter sets to `experiments.py`, commit, and create a pull request (PR).
4. **Contribute**: Open issues or PRs for improvements (e.g., new features, bug fixes). Describe changes clearly (e.g., “Added LR 1e-5 for better F1 score”).
5. **Share Results**: Zip and share `mlruns` folder or upload to a shared drive.


## Example Command

```bash
!python main.py --batch 32 --lr 2e-5 --epochs 20 --author_name 'Nabil' --dataset_path '/content/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data/data/HateSpeech.csv' --freeze_base --mlflow_experiment_name 'Bangla-HateSpeech-Experiments'
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or contributions, open an issue or PR on [GitHub](https://github.com/AnnNaserNabil/Finetune-Bangla-BERT-on-Bangla-Hate-Speech-Data).
