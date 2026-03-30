from datasets import load_dataset
from src.logger import get_logger

logger = get_logger(__name__)

def load_amazon_reviews(
    train_size: int = 5000,
    test_size:  int = 1000,
    seed:       int = 42
):
    """
    Loads the Amazon Polarity dataset from HuggingFace
    and returns shuffled train/test subsets.

    Args:
        train_size : number of training samples
        test_size  : number of test samples
        seed       : random seed for reproducibility

    Returns:
        train_data, test_data (HuggingFace Dataset objects)
    """

    logger.info("Loading amazon_polarity dataset...")

    dataset = load_dataset(
        "amazon_polarity"
    )

    logger.info(f"Full dataset loaded:")
    logger.info(f"  Train split : {len(dataset['train']):,} samples")
    logger.info(f"  Test split  : {len(dataset['test']):,} samples")

    # ── Shuffle and select subsets ───────────────
    logger.info(f"Sampling {train_size:,} train / {test_size:,} test...")

    train_data = dataset['train']\
        .shuffle(seed=seed)\
        .select(range(train_size))

    test_data = dataset['test']\
        .shuffle(seed=seed)\
        .select(range(test_size))

    logger.info("Done! Data ready.")

    return train_data, test_data


def get_sample_reviews(n: int = 3):
    """
    Returns n raw sample reviews for quick inspection.

    Args:
        n : number of samples to return

    Returns:
        list of dicts with title, content, label
    """
    logger.info(f"Fetching {n} sample reviews...")

    dataset = load_dataset(
        "amazon_polarity"
    )

    samples = dataset['train'].select(range(n))

    return [
        {
            'title'  : s['title'],
            'content': s['content'],
            'label'  : 'POSITIVE' if s['label'] == 1 else 'NEGATIVE'
        }
        for s in samples
    ]


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    # Test 1 — load data
    train_data, test_data = load_amazon_reviews(
        train_size=100,   # small for quick test
        test_size=20
    )
    print(f"\nTrain size : {len(train_data)}")
    print(f"Test size  : {len(test_data)}")
    print(f"Features   : {train_data.features}")

    # Test 2 — inspect samples
    samples = get_sample_reviews(n=2)
    print("\nSample Reviews:")
    print("-"*50)
    for s in samples:
        print(f"Title  : {s['title']}")
        print(f"Content: {s['content'][:100]}...")
        print(f"Label  : {s['label']}")
        print()