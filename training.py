import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }

NEW_MODEL_NAME = "financial-trainer"

dataset = load_dataset("takala/financial_phrasebank", "sentences_50agree", trust_remote_code=True)
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(NEW_MODEL_NAME, eval_strategy="epoch")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

print("\nPer-epoch evaluation:")
for entry in trainer.state.log_history:
    if "eval_loss" not in entry:
        continue
    print(
        f"  epoch {entry['epoch']}: "
        f"loss={entry['eval_loss']:.4f} "
        f"accuracy={entry['eval_accuracy']:.4f} "
        f"f1_macro={entry['eval_f1_macro']:.4f} "
        f"f1_weighted={entry['eval_f1_weighted']:.4f}"
    )