# sentiment_analyzer/analyzer.py

from transformers import pipeline, Pipeline
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

class SentimentAnalyzer:
    """
    A class to handle sentiment analysis using a specified Hugging Face model.
    Loads the model pipeline upon initialization.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the SentimentAnalyzer by loading the specified model pipeline.

        Args:
            model_name (str): The name or path of the Hugging Face model to use for sentiment analysis.
        """
        self.model_name = model_name
        self.pipeline: Pipeline | None = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Loads the Hugging Face sentiment analysis pipeline."""
        try:
            logging.info(f"Loading sentiment analysis pipeline for model: '{self.model_name}'...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0
            )
            logging.info(f"Pipeline loaded successfully for model: '{self.model_name}'.")
        except Exception as e:
            logging.error(f"Failed to load pipeline for model '{self.model_name}': {e}", exc_info=True)

    def analyze(self, text_list: list[str]) -> list:
        """
        Analyzes the sentiment of a list of texts using the pre-loaded pipeline.

        Args:
            text_list (list): A list of strings, where each string is a text to analyze.

        Returns:
            list: A list of dictionaries containing 'label' and 'score' for each input text.
                  Returns an empty list if the pipeline isn't loaded or an error occurs.
                  Returns an empty list if the input is not a non-empty list.
        """
        if self.pipeline is None:
            logging.error(f"Sentiment pipeline for model '{self.model_name}' is not available.")
            return []

        # Input validation
        if not isinstance(text_list, list):
            logging.warning("Input is not a list. Please provide a list of strings.")
            return []
        if not text_list:
            logging.warning("Input text list is empty.")
            return []
        if not all(isinstance(text, str) for text in text_list):
            logging.warning("Input list contains non-string elements.")
            return []

        try:
            logging.info(f"Analyzing sentiment for {len(text_list)} text(s) using model '{self.model_name}'...")
            results = self.pipeline(text_list)
            logging.info("Sentiment analysis complete.")
            return results
        except Exception as e:
            logging.error(f"Error during sentiment analysis with model '{self.model_name}': {e}", exc_info=True)
            return []