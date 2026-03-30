from transformers import AutoTokenizer
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)


def prepare_examples(example):
    """
    Combines title + content into single text field.

    Args:
        example : single dataset row

    Returns:
        dict with 'text' and 'label' fields
    """
    return {
        'text' : example['title'] + " " + example['content'],
        'label': example['label']
    }


def tokenize(examples):
    """
    Tokenizes text field using DistilBERT tokenizer.

    Args:
        examples : batch of examples

    Returns:
        tokenized batch with input_ids, attention_mask
    """
    return tokenizer(
        examples['text'],
        truncation = True,
        max_length = 256
    )


def preprocess(train_data, test_data):
    """
    Full preprocessing pipeline:
    1. Combine title + content
    2. Remove unused columns
    3. Tokenize

    Args:
        train_data : raw HuggingFace train dataset
        test_data  : raw HuggingFace test dataset

    Returns:
        tokenized_train, tokenized_test
    """

    logger.info("Starting preprocessing...")

    # ── Step 1: Combine title + content ─────────
    logger.info("Combining title + content...")
    train_data = train_data.map(prepare_examples)
    test_data  = test_data.map(prepare_examples)

    # ── Step 2: Remove unused columns ───────────
    logger.info("Removing unused columns...")
    train_data = train_data.remove_columns(['title', 'content'])
    test_data  = test_data.remove_columns(['title', 'content'])

    # ── Step 3: Tokenize ─────────────────────────
    logger.info("Tokenizing...")
    tokenized_train = train_data.map(tokenize, batched=True)
    tokenized_test  = test_data.map(tokenize,  batched=True)

        # ── Step 4: Remove text column ───────────────
    # Trainer can't handle raw text strings in tensors
    tokenized_train = tokenized_train.remove_columns(['text'])
    tokenized_test  = tokenized_test.remove_columns(['text'])

    # ── Step 5: Set format to PyTorch ────────────
    tokenized_train.set_format("torch")

    logger.info(f"Preprocessing complete!")
    logger.info(f"  Train features: {tokenized_train.features}")
    logger.info(f"  Train size    : {len(tokenized_train):,}")
    logger.info(f"  Test size     : {len(tokenized_test):,}")

    return tokenized_train, tokenized_test


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_amazon_reviews

    # Load small sample for quick test
    train_data, test_data = load_amazon_reviews(
        train_size=100,
        test_size=20
    )

    # Preprocess
    tokenized_train, tokenized_test = preprocess(
        train_data, test_data
    )

    # Inspect one tokenized example
    print("\nSample tokenized example:")
    print("-"*50)
    print(f"Label      : {tokenized_train[0]['label']}")
    print(f"Input IDs  : {tokenized_train[0]['input_ids'][:10]}...")
    print(f"Attn Mask  : {tokenized_train[0]['attention_mask'][:10]}...")
    print(f"\nLabel set  : {set(tokenized_train['label'])}")