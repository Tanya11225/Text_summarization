from transformers import pipeline

# Load a pre-trained summarizer model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    if len(text) < 100:
        return "Text too short to summarize."

    # Limit input for model safety (1000 characters)
    input_text = text[:1000]

    try:
        summary = summarizer(input_text, max_length=150, min_length=140, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {e}"

# New function to generate a summary between 500 and 1000 words
def generate_long_summary(text):
    if len(text) < 100:
        return "Text too short to summarize."

    # For long summaries, use more of the input text (up to 4000 characters)
    input_text = text[:4000]

    try:
        # Set min_length and max_length for 500-1000 words (approx 3500-7000 characters)
        # Since the model works in tokens, we estimate: 1 word ≈ 1.3 tokens
        # 500 words ≈ 650 tokens, 1000 words ≈ 1300 tokens
        summary = summarizer(
            input_text,
            min_length=650,  # ~500 words
            max_length=1300, # ~1000 words
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Long summarization error: {e}"
