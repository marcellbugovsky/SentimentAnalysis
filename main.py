# main.py

import logging

from sentiment_analyzer.analyzer import analyze_sentiment

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Input ---
example_texts = [
    "This is a fantastic library! I love using it.",
    "The documentation could be clearer, and I encountered some bugs.",
    "It's an okay tool, neither good nor bad.",
    "This is just a neutral statement.",
    "", # Test empty string
    None # Test None value (should be handled by input validation ideally)
]

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("Starting sentiment analysis process...")

    # Filter out potential None values or other non-strings before sending to analyzer
    valid_texts = [text for text in example_texts if isinstance(text, str)]
    if len(valid_texts) != len(example_texts):
        logging.warning("Some invalid input items (e.g., None) were filtered out.")

    results = analyze_sentiment(valid_texts)

    # --- Process and Print results ---
    print("\n--- Results ---")
    if results:
        # Make sure we align results with the valid texts we sent
        for i, result in enumerate(results):
            if i < len(valid_texts):
                text = valid_texts[i]
                label = result.get('label', 'N/A')
                score = result.get('score', 0.0)
                # Handle potentially empty text display
                display_text = text if text else "'' (Empty String)"
                print(f"Text: \"{display_text}\"")
                print(f"Sentiment: {label}, Score: {score:.4f}\n")
            else:
                logging.warning(f"Result index {i} out of bounds for valid input texts.")
    else:
        # Check if the input itself was empty after filtering
        if not valid_texts:
            print("No valid texts provided for analysis.")
        else:
            print("No results returned from analysis (check logs for errors).")

    print("--- Analysis Complete ---")