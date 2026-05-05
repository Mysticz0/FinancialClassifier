# Financial Classifier

This project trains a financial sentiment classifier with Hugging Face Transformers and uses it to score the sentiment of finance news articles from a URL.

Sentiment labels:
- `0`: Negative
- `1`: Neutral
- `2`: Positive

## Project Files

- `training.py`: fine-tunes one or more Transformer models on Financial PhraseBank and logs runs to MLflow.
- `main.py`: interactive CLI that downloads an article, splits it into sentences, classifies each sentence, and prints an overall sentiment.
- `mlflow.db`: local MLflow SQLite tracking database (generated/updated during training).

## Dataset

Training uses [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank), configuration `sentences_50agree`, then creates an 80/20 train-validation split with `seed=42`.

## Setup

Use Python 3.9+.

Install core dependencies:

```bash
pip install torch transformers datasets scikit-learn numpy accelerate mlflow nltk requests beautifulsoup4
```

For GPU support, install the appropriate PyTorch build for your system from [pytorch.org](https://pytorch.org/get-started/locally/).

## Training

Run:

```bash
python training.py
```

What `training.py` does:
- Loads Financial PhraseBank and tokenizes the `sentence` field.
- Trains each model listed in `models_to_train` (currently `roberta-base`).
- Evaluates each epoch with accuracy, macro F1, and weighted F1.
- Uses early stopping (`patience=2`) and loads the best checkpoint at the end.
- Logs params/metrics to MLflow (`sqlite:///mlflow.db`).
- Saves checkpoints under `BASE_DIR/<model-name>/checkpoint-*`.

Important:
- Update `BASE_DIR` in `training.py` to a valid path on your machine before training.

## Running Inference

Run:

```bash
python main.py
```

Workflow:
- Enter a news article URL.
- The script extracts paragraph text from the page.
- The text is split into sentences (`nltk.sent_tokenize`).
- Each sentence is scored by the model `Mysticz0/finance-pro-model-v1.0`.
- Sentence logits are accumulated to produce a final article-level sentiment.

If NLTK tokenizer data is missing, the script downloads `punkt_tab` automatically.

## MLflow

To inspect experiment runs locally:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Notes

- `main.py` currently uses a hosted Hugging Face model (`Mysticz0/finance-pro-model-v1.0`) for inference rather than loading directly from local checkpoints.
- First run requires internet access to download the dataset/model artifacts.
