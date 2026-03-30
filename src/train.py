import os
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from src.logger import get_logger
from src.preprocessing import tokenizer

logger = get_logger(__name__)

MODEL_NAME  = "distilbert-base-uncased"
OUTPUT_DIR  = "./models/sentiment-model"


def get_model():
    """
    Loads DistilBERT with 2-class classification head.

    Returns:
        model with NEGATIVE/POSITIVE label config
    """
    logger.info(f"Loading {MODEL_NAME}...")

    model = AutoModelForSequenceClassification\
        .from_pretrained(MODEL_NAME, num_labels=2)

    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    logger.info("Model loaded!")
    return model


def get_compute_metrics():
    """
    Returns compute_metrics function for Trainer.
    Calculates accuracy and F1 score.
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


def train(tokenized_train, tokenized_test):
    """
    Full training pipeline:
    1. Load model
    2. Set training arguments
    3. Train with Trainer API
    4. Evaluate
    5. Save model locally

    Args:
        tokenized_train : preprocessed train dataset
        tokenized_test  : preprocessed test dataset

    Returns:
        trainer, results
    """

    # ── Model ────────────────────────────────────
    model = get_model()

    # ── Training Arguments ───────────────────────
    logger.info("Setting up training arguments...")

    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 3,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-5,
        warmup_steps                = 100,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        logging_steps               = 50,
        seed                        = 42,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer)

    # ── Trainer ──────────────────────────────────
    trainer = Trainer(
        model            = model,
        args             = args,
        train_dataset    = tokenized_train,
        eval_dataset     = tokenized_test,
        processing_class = tokenizer,
        data_collator    = data_collator,
        compute_metrics  = get_compute_metrics(),
    )

    # ── Train ────────────────────────────────────
    logger.info("Starting training...")
    logger.info(f"  Train samples : {len(tokenized_train):,}")
    logger.info(f"  Test samples  : {len(tokenized_test):,}")
    logger.info(f"  Epochs        : {args.num_train_epochs}")
    logger.info(f"  Batch size    : {args.per_device_train_batch_size}")

    trainer.train()

    # ── Evaluate ─────────────────────────────────
    logger.info("Evaluating...")
    results = trainer.evaluate()

    logger.info("="*45)
    logger.info("RESULTS")
    logger.info("="*45)
    logger.info(f"  Accuracy : {results['eval_accuracy']:.4f} "
                f"({results['eval_accuracy']*100:.2f}%)")
    logger.info(f"  F1 Score : {results['eval_f1']:.4f}")
    logger.info(f"  Loss     : {results['eval_loss']:.4f}")

    # ── Save locally ─────────────────────────────
    logger.info(f"Saving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Model saved!")

    return trainer, results


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_amazon_reviews
    from src.preprocessing import preprocess

    # Small sample — just to verify pipeline works
    # Use full 5000/1000 for real training on Colab
    logger.info("Loading data...")
    train_data, test_data = load_amazon_reviews(
        train_size=500,
        test_size=100
    )

    logger.info("Preprocessing...")
    tokenized_train, tokenized_test = preprocess(
        train_data, test_data
    )

    # Train
    trainer, results = train(tokenized_train, tokenized_test)

    print("\nTraining complete!")
    print(f"Accuracy : {results['eval_accuracy']*100:.2f}%")
    print(f"F1 Score : {results['eval_f1']:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")