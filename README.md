# Financial classifier

Fine-tunes [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on the [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (`sentences_50agree` split) for **three-class sentiment**: negative (0), neutral (1), positive (2).

## Repository layout

| Path | Description |
|------|-------------|
| `training.py` | Loads data, fine-tunes FinBERT, and **writes the output directory** `financial-trainer/` (see `NEW_MODEL_NAME`). |
| `financial-trainer/` | **Generated artifact** — created when you run `training.py`. Contains Hugging Face `Trainer` checkpoints (e.g. `checkpoint-500`, `checkpoint-1000`, …). Listed in `.gitignore` so it is not committed; regenerate locally after clone. |
| `example_usage.py` | Loads checkpoints under `financial-trainer/`, splits a sample article into sentences with **NLTK** `sent_tokenize` (handles abbreviations such as `U.S.`), runs the classifier per sentence, and sums logits for an overall document-style label. |

## Setup

Python 3.9+ is recommended. Install dependencies (versions may vary with your environment):

```bash
pip install torch transformers datasets scikit-learn numpy accelerate nltk
```

It is recommended to be using a GPU for training. Install `torchvision` using website command at https://pytorch.org/get-started/locally/. Example for Windows:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

You need network access on first run to download the dataset and the base model from Hugging Face.

The first time sentence splitting runs, NLTK may download the `punkt_tab` tokenizer data automatically (see `example_usage.py`).

## Train (generates `financial-trainer/`)

From this directory:

```bash
python training.py
```

Training uses:

- **Dataset:** `takala/financial_phrasebank`, config `sentences_50agree`, with an 80/20 train/validation split (`seed=42`).
- **Output directory:** `financial-trainer` (see `NEW_MODEL_NAME` in `training.py`).
- **Metrics:** accuracy and macro/weighted F1, evaluated each epoch.

After training finishes, checkpoints appear under `financial-trainer/checkpoint-*`. The exact step numbers depend on dataset size and training configuration.

## Run inference (`example_usage.py`)

```bash
python example_usage.py
```

- Set `SELECTED_MODEL_INDEX` to choose which of the listed checkpoint paths to use (paths must match folders inside `financial-trainer/` after training).
- The embedded sample `article` is normalized to whitespace, split with `nltk.tokenize.sent_tokenize`, then each sentence is classified; logits are accumulated and an overall label is printed.

## Notes

- The repo does not include `financial-trainer/` in version control. After cloning, run `python training.py` before `example_usage.py` (unless you point inference at another checkpoint path).
- If `financial-trainer/` is missing or you change hyperparameters, run `training.py` again; checkpoint paths and `SELECTED_MODEL_INDEX` in `example_usage.py` may need updating to match new runs.
- Uncommenting `selected_model.push_to_hub(...)` at the bottom of `example_usage.py` would upload a checkpoint to the Hugging Face Hub (requires authentication and a chosen repo id).
