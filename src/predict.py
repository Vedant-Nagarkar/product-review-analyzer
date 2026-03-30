import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)
from src.logger import get_logger

logger = get_logger(__name__)

# ── Device ────────────────────────────────────
DEVICE     = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model paths ───────────────────────────────
SENTIMENT_MODEL_PATH = "./models/sentiment-model"

# ── Categories & Aspects ──────────────────────
CATEGORIES = [
    "Electronics",
    "Clothing and Fashion",
    "Books and Literature",
    "Food and Grocery",
    "Sports and Outdoors",
    "Home and Kitchen",
    "Beauty and Personal Care",
    "Toys and Games"
]

ASPECTS = [
    "price and value",
    "quality and durability",
    "delivery and shipping",
    "customer service",
    "ease of use",
    "design and appearance",
    "performance and speed",
    "battery life"
]


def load_models():
    """
    Loads all 4 models once at startup.
    Returns dict of all loaded pipelines/models.
    """
    logger.info("Loading all models...")
    models = {}

    # ── 1. Sentiment (fine-tuned DistilBERT) ────
    logger.info("Loading sentiment model...")
    models['sentiment'] = pipeline(
        "text-classification",
        model     = SENTIMENT_MODEL_PATH,
        tokenizer = SENTIMENT_MODEL_PATH,
        device    = DEVICE
    )
    logger.info("Sentiment model loaded!")

    # ── 2. Category (BART-MNLI zero-shot) ───────
    logger.info("Loading category classifier...")
    models['category'] = pipeline(
        "zero-shot-classification",
        model  = "facebook/bart-large-mnli",
        device = DEVICE
    )
    logger.info("Category classifier loaded!")

    # ── 3. Aspect (RoBERTa NLI zero-shot) ───────
    logger.info("Loading aspect analyzer...")
    models['aspect'] = pipeline(
        "zero-shot-classification",
        model  = "cross-encoder/nli-roberta-base",
        device = DEVICE
    )
    logger.info("Aspect analyzer loaded!")

    # ── 4. Summarization (BART-XSUM) ────────────
    logger.info("Loading summarization model...")
    models['bart_tokenizer'] = BartTokenizer.from_pretrained(
        "facebook/bart-large-xsum")
    models['bart_model'] = BartForConditionalGeneration\
        .from_pretrained("facebook/bart-large-xsum")\
        .to(DEVICE_STR)
    logger.info("Summarization model loaded!")

    logger.info("All models loaded successfully!")
    return models


# ── Load once at module level ─────────────────
# This means models load once when predict.py
# is imported, not on every function call
MODELS = load_models()


def analyze_sentiment(text: str) -> dict:
    """
    Predicts sentiment of review text.

    Args:
        text : review text

    Returns:
        dict with label (POSITIVE/NEGATIVE) and score
    """
    result = MODELS['sentiment'](text[:512])[0]
    return {
        'label': result['label'],
        'score': round(result['score'], 4)
    }


def classify_category(text: str) -> dict:
    """
    Classifies review into product category
    using zero-shot classification.

    Args:
        text : review text

    Returns:
        dict with category and confidence score
    """
    result = MODELS['category'](
        text[:512],
        candidate_labels = CATEGORIES,
        multi_label      = False
    )
    return {
        'category': result['labels'][0],
        'score'   : round(result['scores'][0], 4)
    }


def analyze_aspects(text: str) -> list:
    """
    Identifies top aspects mentioned in review
    using zero-shot classification.

    Args:
        text : review text

    Returns:
        list of (aspect, score) tuples, top 3
        with score > 0.3
    """
    result = MODELS['aspect'](
        text[:512],
        candidate_labels = ASPECTS,
        multi_label      = True
    )
    return [
        (label, round(score, 4))
        for label, score in zip(
            result['labels'],
            result['scores']
        )
        if score > 0.3
    ][:3]


def summarize_review(text: str) -> str:
    """
    Summarizes review using BART-XSUM.
    Short reviews returned as-is.

    Args:
        text : review text

    Returns:
        concise summary string
    """
    if len(text.split()) < 30:
        return text

    inputs = MODELS['bart_tokenizer'](
        text[:512],
        return_tensors = "pt",
        truncation     = True,
        max_length     = 512
    ).to(DEVICE_STR)

    summary_ids = MODELS['bart_model'].generate(
        inputs["input_ids"],
        max_new_tokens = 80,
        min_length     = 15,
        length_penalty = 2.0,
        num_beams      = 4,
        early_stopping = True
    )

    return MODELS['bart_tokenizer'].decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def analyze_review(review_text: str) -> dict:
    """
    Master function — runs all 4 models on
    a single review and returns complete analysis.

    Args:
        review_text : full review text

    Returns:
        dict with sentiment, category, aspects, summary
    """
    logger.info("Analyzing review...")

    sentiment = analyze_sentiment(review_text)
    logger.info(f"Sentiment : {sentiment['label']} "
                f"({sentiment['score']})")

    category  = classify_category(review_text)
    logger.info(f"Category  : {category['category']} "
                f"({category['score']})")

    aspects   = analyze_aspects(review_text)
    logger.info(f"Aspects   : {[a[0] for a in aspects]}")

    summary   = summarize_review(review_text)
    logger.info(f"Summary   : {summary[:60]}...")

    return {
        'sentiment': sentiment,
        'category' : category,
        'aspects'  : aspects,
        'summary'  : summary
    }


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    test_reviews = [
        """This laptop is absolutely incredible. Battery lasts 
        all day, easily 10-12 hours of real work. The display 
        is crisp and bright. Performance is blazing fast. 
        Highly recommend!""",

        """Complete waste of money. Stopped working after a week.
        Customer service was useless and refused a refund.
        Packaging was damaged too. Avoid at all costs.""",

        """Ordered these running shoes for marathon training.
        Delivery was super fast, arrived in 2 days. Cushioning
        is excellent. Only downside is sizing runs small,
        order a size up."""
    ]

    print("\n" + "="*55)
    for i, review in enumerate(test_reviews):
        print(f"\nREVIEW {i+1}")
        print("-"*55)
        result = analyze_review(review)

        emoji = "😊" if result['sentiment']['label'] \
                == "POSITIVE" else "😞"

        print(f"{emoji} Sentiment : {result['sentiment']['label']} "
            f"({result['sentiment']['score']})")
        print(f"📦 Category : {result['category']['category']} "
            f"({result['category']['score']})")
        print(f"🔍 Aspects  :")
        for aspect, score in result['aspects']:
            print(f"   → {aspect} ({score})")
        print(f"📝 Summary  : {result['summary']}")
        print("="*55)