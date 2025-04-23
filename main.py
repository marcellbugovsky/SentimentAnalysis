# main.py

import argparse
import logging

from sentiment_analyzer.analyzer import SentimentAnalyzer, DEFAULT_MODEL

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to parse arguments and perform sentiment analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze sentiment of a given text using Hugging Face models.")

    # Argument for the input text
    parser.add_argument("-t", "--text",
                        type=str,
                        required=True, # Make text input mandatory
                        help="The text string to analyze.")

    # Argument for the model name
    parser.add_argument("-m", "--model",
                        type=str,
                        default=DEFAULT_MODEL, # Use the default from analyzer module
                        help=f"The Hugging Face model name for sentiment analysis (default: {DEFAULT_MODEL}).")

    args = parser.parse_args()

    # --- Instantiate Analyzer and Analyze ---
    logging.info(f"Initializing analyzer with model: {args.model}")
    # Create an instance of our analyzer class with the specified model
    analyzer = SentimentAnalyzer(model_name=args.model)

    # The analyzer expects a list of texts
    input_text_list = [args.text]

    # Perform the analysis
    results = analyzer.analyze(input_text_list)

    # --- Process and Print Results ---
    print("\n--- Result ---")
    if results:
        # Since we analyze only one text, we expect only one result
        result = results[0]
        label = result.get('label', 'N/A')
        score = result.get('score', 0.0)
        print(f"Text: \"{args.text}\"")
        print(f"Sentiment: {label}")
        print(f"Confidence Score: {score:.4f}\n")
    else:
        print(f"Could not analyze the text (check logs for errors using model: {args.model}).")

    print("--- Analysis Complete ---")


if __name__ == "__main__":
    main()