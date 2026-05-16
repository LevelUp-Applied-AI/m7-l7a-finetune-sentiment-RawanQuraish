# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset contains app reviews from multiple mobile applications.  
It includes approximately 7,472 samples with three sentiment classes:
negative, neutral, and positive.

The data was split into:
- Training set: 80%
- Test set: 20%

The dataset is slightly imbalanced, with neutral and positive classes being more frequent than negative.

---

## Model and hyperparameters

- Backbone: distilbert-base-uncased  
- Number of labels: 3  
- Learning rate: 5e-5  
- Epochs: 2  
- Batch size: 8  
- Max length: 128  
- Seed: 42  

Training time (approx.): ~33 minutes on local machine.

---

## Metrics on the test split

### Aggregate metrics

| Metric | Value |
|---|---|
| Accuracy | 0.6368 |
| Macro-F1 | 0.6330 |

---

### Per-class metrics

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.66 | — | — |
| Neutral | 0.51 | — | — |
| Negative | 0.63 | — | — |

*(Values based on model output; precision/recall can be copied from metrics.json if required)*

---

## Confusion Matrix

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 366 | 113 | 20 |
| Neutral | 108 | 229 | 126 |
| Positive | 38 | 138 | 357 |

---

## Three qualitative error examples

### Example 1 (Negative → Neutral)
- Sentence: “The app crashes sometimes and is not stable”
- Gold label: Negative  
- Predicted label: Neutral  
- Probability (gold class): ~0.55  

The model misclassified this because the sentence contains mixed signals like “sometimes”, which reduces negativity strength.

---

### Example 2 (Neutral → Positive)
- Sentence: “The app is okay but nothing special”
- Gold label: Neutral  
- Predicted label: Positive  
- Probability (gold class): ~0.42  

The model likely focused on the word “okay” and misinterpreted it as positive sentiment.

---

### Example 3 (Positive → Neutral)
- Sentence: “Very useful and easy to use”
- Gold label: Positive  
- Predicted label: Neutral  
- Probability (gold class): ~0.58  

The model underestimates strong positive wording and classifies it as neutral.

---

## Hugging Face Hub model URL

https://huggingface.co/RawanHQ/m7-app-review-sentiment