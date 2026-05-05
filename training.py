import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
import mlflow

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

BASE_DIR = "W:/Programming/Python/ml-checkpoints"

#"bert-base-uncased", "ProsusAI/finbert", "distilbert-base-uncased", "roberta-base", "albert-base-v2"
models_to_train = ["roberta-base"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("financial-classifier")

dataset = load_dataset("takala/financial_phrasebank", "sentences_50agree", trust_remote_code=True)
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

for model in models_to_train:

    run_name = f"{model.replace('/', '-')}"
    tokenizer = AutoTokenizer.from_pretrained(model)
    current_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        f"{BASE_DIR}/{run_name}", 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        num_train_epochs=10, 
        load_best_model_at_end=True, 
        metric_for_best_model="f1_macro", 
        greater_is_better=True,
        report_to="mlflow",
    )

    print(f"--------------------------------Training {model}--------------------------------")

    with mlflow.start_run(run_name=model):
        mlflow.log_param("model_name", model)
        mlflow.log_param("num_labels", 3)
        mlflow.log_param("train_size", len(tokenized_datasets["train"]))
        mlflow.log_param("validation_size", len(tokenized_datasets["validation"]))
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("num_epochs", training_args.num_train_epochs)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("weight_decay", training_args.weight_decay)
        
        trainer = Trainer(
            model=current_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()

    print(f"--------------------------------Per-Epoch Evaluation {model}--------------------------------")
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