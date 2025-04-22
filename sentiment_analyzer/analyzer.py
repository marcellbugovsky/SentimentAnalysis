# sentiment_analyzer/analyzer.py

from transformers import pipeline
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Model ---
# Load the pipeline once when the module is loaded for efficiency.
SENTIMENT_PIPELINE = None # Initialized as None
try:
    logging.info("Loading sentiment analysis pipeline (this may take a moment)...")
    SENTIMENT_PIPELINE = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0
    )
    logging.info("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    # Log the error and keep SENTIMENT_PIPELINE as None
    logging.error(f"Failed to load sentiment analysis pipeline: {e}", exc_info=True)

# --- Analysis Function ---
def analyze_sentiment(text_list):
    """
    Analyzes the sentiment of a list of texts using the pre-loaded pipeline.

    Args:
        text_list (list): A list of strings, where each string is a text to analyze.

    Returns:
        list: A list of dictionaries containing 'label' and 'score' for each input text.
              Returns an empty list if the pipeline isn't loaded or an error occurs.
              Returns an empty list if the input is not a non-empty list.
    """
    global SENTIMENT_PIPELINE

    if SENTIMENT_PIPELINE is None:
        logging.error("Sentiment pipeline is not available.")
        return []

    # Input validation
    if not isinstance(text_list, list):
        logging.warning("Input is not a list. Please provide a list of strings.")
        return []
    if not text_list:
        logging.warning("Input text list is empty.")
        return []

    try:
        logging.info(f"Analyzing sentiment for {len(text_list)} text(s)...")
        results = SENTIMENT_PIPELINE(text_list)
        logging.info("Sentiment analysis complete.")
        return results
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return []