import numpy as np
import evaluate
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "./models/sentiment-model"


def load_trained_model():
    """
    Loads the fine-tuned sentiment model and tokenizer
    from local models/ folder.

    Returns:
        model, tokenizer
    """
    logger.info(f"Loading trained model from {MODEL_PATH}...")

    model     = AutoModelForSequenceClassification\
        .from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer\
        .from_pretrained(MODEL_PATH)

    logger.info("Model loaded successfully!")
    return model, tokenizer


def get_compute_metrics():
    """
    Returns compute_metrics function with
    accuracy and F1 score.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions    = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_metric.compute(
                predictions=predictions,
                references=labels)['accuracy'],
            'f1': f1_metric.compute(
                predictions=predictions,
                references=labels,
                average='weighted')['f1']
        }

    return compute_metrics


def evaluate_model(tokenized_test):
    """
    Runs full evaluation on test dataset.

    Args:
        tokenized_test : preprocessed test dataset

    Returns:
        results dict with accuracy, f1, loss
    """
    model, tokenizer = load_trained_model()

    # ── Trainer just for evaluation ──────────────
    args = TrainingArguments(
        output_dir     = "./models/eval-output",
        per_device_eval_batch_size = 32,
        seed           = 42,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer)

    trainer = Trainer(
        model           = model,
        args            = args,
        eval_dataset    = tokenized_test,
        processing_class= tokenizer,
        data_collator   = data_collator,
        compute_metrics = get_compute_metrics(),
    )

    # ── Evaluate ─────────────────────────────────
    logger.info("Running evaluation...")
    results = trainer.evaluate()

    # ── Log results ──────────────────────────────
    logger.info("="*45)
    logger.info("EVALUATION RESULTS")
    logger.info("="*45)
    logger.info(f"  Accuracy : {results['eval_accuracy']:.4f} "
                f"({results['eval_accuracy']*100:.2f}%)")
    logger.info(f"  F1 Score : {results['eval_f1']:.4f}")
    logger.info(f"  Loss     : {results['eval_loss']:.4f}")

    return results


def plot_confusion_matrix(tokenized_test):
    """
    Generates and saves confusion matrix plot
    to outputs/ folder.

    Args:
        tokenized_test : preprocessed test dataset
    """
    import torch
    from torch.utils.data import DataLoader

    model, tokenizer = load_trained_model()
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader    = DataLoader(
        tokenized_test,
        batch_size    = 32,
        collate_fn    = data_collator
    )

    all_preds  = []
    all_labels = []

    logger.info("Generating predictions for confusion matrix...")

    import torch
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids      = batch['input_ids'],
                attention_mask = batch['attention_mask']
            )
            preds  = np.argmax(
                outputs.logits.numpy(), axis=-1)
            labels = batch['labels'].numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # ── Plot ─────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = 'd',
        cmap       = 'Blues',
        xticklabels= ['NEGATIVE', 'POSITIVE'],
        yticklabels= ['NEGATIVE', 'POSITIVE']
    )
    plt.title('Confusion Matrix — Sentiment Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    logger.info("Confusion matrix saved to outputs/confusion_matrix.png")
    plt.show()


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_amazon_reviews
    from src.preprocessing import preprocess

    # Load small test sample
    _, test_data = load_amazon_reviews(
        train_size=100,
        test_size=100
    )

    _, tokenized_test = preprocess(
        test_data, test_data  # passing test as both, we only need test
    )

    # Evaluate
    results = evaluate_model(tokenized_test)

    print("\nEvaluation complete!")
    print(f"Accuracy : {results['eval_accuracy']*100:.2f}%")
    print(f"F1 Score : {results['eval_f1']:.4f}")

    # Confusion matrix
    plot_confusion_matrix(tokenized_test)
    print("Confusion matrix saved to outputs/")