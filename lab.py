"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment
"""

import json
import os
import numpy as np
import pandas as pd
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}



def get_data_path() -> str:
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    split = dataset.train_test_split(test_size=test_size, seed=seed)

    return DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    return ds_dict.map(tokenize_fn, batched=True)



def make_training_args(output_dir: str, lr=5e-5, epochs=2, batch_size=8, seed=42):
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    # 🔥 FIX مهم جداً للاختبارات
    args.eval_strategy = "epoch"
    args.save_strategy = "epoch"

    return args



def compute_metrics(eval_pred):
    logits, labels = eval_pred  

    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }



def train_classifier(
    tokenized_ds,
    model_name,
    training_args,
    tokenizer,
    num_labels=3,
):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer



def evaluate_classifier(trainer, tokenized_test):

    output = trainer.predict(tokenized_test)
    logits = output.predictions
    labels = output.label_ids

    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    f1_per_class = f1_score(labels, preds, average=None)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    id2label = trainer.model.config.id2label

    per_class_f1 = {id2label[i]: float(f1_per_class[i]) for i in range(len(id2label))}
    per_class_precision = {id2label[i]: float(precision_per_class[i]) for i in range(len(id2label))}
    per_class_recall = {id2label[i]: float(recall_per_class[i]) for i in range(len(id2label))}

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall
    }



def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)



def main():

    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)

    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(output_dir)

    trainer = train_classifier(
        tokenized,
        model_name,
        training_args,
        tokenizer
    )

    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    
    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    
    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    probs = _softmax(pred_logits)

    id2label = trainer.model.config.id2label

    
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=["negative", "neutral", "positive"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["negative", "neutral", "positive"],
        columns=["negative", "neutral", "positive"]
    )

    cm_df.to_csv("confusion_matrix.csv")

    
    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(probs[i][pred_idx[i]]) for i in range(len(pred_idx))],

        "prob_negative": [float(probs[i][0]) for i in range(len(pred_idx))],
        "prob_neutral": [float(probs[i][1]) for i in range(len(pred_idx))],
        "prob_positive": [float(probs[i][2]) for i in range(len(pred_idx))]
    })

    df_out.to_csv("predictions.csv", index=False)

    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"Pushed to HF Hub: https://huggingface.co/RawanHQ/{repo_id}")
        except Exception as e:
            print("HF push failed:", e)


if __name__ == "__main__":
    main()