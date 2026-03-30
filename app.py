import gradio as gr
from src.predict import analyze_review
from src.logger import get_logger

logger = get_logger(__name__)


def run_analysis(review_text: str):
    """
    Wrapper function for Gradio interface.
    Takes review text, returns formatted outputs
    for each Gradio component.

    Args:
        review_text : raw review input from user

    Returns:
        sentiment, category, aspects, summary
        (formatted for Gradio components)
    """
    if not review_text.strip():
        return (
            "Please enter a review!",
            "Please enter a review!",
            "Please enter a review!",
            "Please enter a review!"
        )

    logger.info("Running analysis from Gradio UI...")

    result = analyze_review(review_text)

    # ── Format Sentiment ─────────────────────────
    emoji     = "😊" if result['sentiment']['label'] \
                == "POSITIVE" else "😞"
    sentiment = (
        f"{emoji} {result['sentiment']['label']}\n"
        f"Confidence: {result['sentiment']['score']*100:.1f}%"
    )

    # ── Format Category ──────────────────────────
    category = (
        f"📦 {result['category']['category']}\n"
        f"Confidence: {result['category']['score']*100:.1f}%"
    )

    # ── Format Aspects ───────────────────────────
    aspects_text = "🔍 Aspects Mentioned:\n\n"
    for aspect, score in result['aspects']:
        bar   = "█" * int(score * 10)
        empty = "░" * (10 - int(score * 10))
        aspects_text += f"• {aspect:<25} {score:.2f}  {bar}{empty}\n"

    if not result['aspects']:
        aspects_text += "No strong aspects detected."

    # ── Format Summary ───────────────────────────
    summary = f"📝 {result['summary']}"

    return sentiment, category, aspects_text, summary


# ── Gradio UI ─────────────────────────────────
def create_ui():

    with gr.Blocks(
        title = "pranalyzer — Product Review Analyzer",
        theme = gr.themes.Soft()
    ) as demo:

        # ── Header ───────────────────────────────
        gr.Markdown("""
        # 🛍️ pranalyzer
        ### Product Review Analyzer
        Paste any Amazon product review below and get instant analysis
        using 4 NLP models running in parallel.
        ---
        """)

        # ── Input ────────────────────────────────
        with gr.Row():
            review_input = gr.Textbox(
                label       = "📝 Paste your product review here",
                placeholder = "e.g. This laptop is amazing! Battery lasts all day...",
                lines       = 6,
                scale       = 4
            )

        # ── Analyze Button ────────────────────────
        analyze_btn = gr.Button(
            "🔍 Analyze Review",
            variant = "primary",
            size    = "lg"
        )

        # ── Output Row ───────────────────────────
        gr.Markdown("### 📊 Analysis Results")

        with gr.Row():
            sentiment_out = gr.Textbox(
                label    = "😊 Sentiment",
                lines    = 3,
                scale    = 1
            )
            category_out  = gr.Textbox(
                label    = "📦 Category",
                lines    = 3,
                scale    = 1
            )

        with gr.Row():
            aspects_out = gr.Textbox(
                label    = "🔍 Aspects",
                lines    = 6,
                scale    = 1
            )
            summary_out = gr.Textbox(
                label    = "📝 Summary",
                lines    = 6,
                scale    = 1
            )

        # ── Example Reviews ──────────────────────
        gr.Markdown("### 💡 Try these examples")
        gr.Examples(
            examples = [
                ["This laptop is absolutely incredible! Battery lasts all day, easily 10-12 hours of real work. The display is crisp and bright, perfect for both indoor and outdoor use. Performance is blazing fast. Highly recommended!"],
                ["Complete waste of money. Stopped working after a week. Customer service was useless and refused a refund. Packaging was damaged too. Avoid at all costs."],
                ["Ordered these running shoes for marathon training. Delivery was super fast, arrived in 2 days. Cushioning is excellent and my feet feel great even after 20km runs. Only downside is sizing runs small, order a size up."],
                ["This cookbook is a disappointment. Half the recipes have missing ingredients. The photos look amazing but actual results look nothing like them. Very misleading. Wasted expensive ingredients trying three different recipes."]
            ],
            inputs = review_input
        )

        # ── Footer ───────────────────────────────
        gr.Markdown("""
        ---
        **Models used:**
        `DistilBERT` → Sentiment &nbsp;|&nbsp;
        `BART-MNLI` → Category &nbsp;|&nbsp;
        `RoBERTa` → Aspects &nbsp;|&nbsp;
        `BART-XSUM` → Summary

        Built by [Vedant Nagarkar](https://huggingface.co/Ved2001) •
        Model: [Ved2001/pranalyzer](https://huggingface.co/Ved2001/pranalyzer)
        """)

        # ── Connect button to function ────────────
        analyze_btn.click(
            fn      = run_analysis,
            inputs  = review_input,
            outputs = [
                sentiment_out,
                category_out,
                aspects_out,
                summary_out
            ]
        )

    return demo


# ── Launch ────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting pranalyzer Gradio app...")
    demo = create_ui()
    demo.launch(
        share        = False,   # True to get public URL
        server_port  = 7860,
        show_error   = True
    )