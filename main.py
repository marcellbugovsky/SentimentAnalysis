# main.py

from transformers import pipeline

# 1. Load the sentiment analysis pipeline
try:
    print("Loading sentiment analysis pipeline...")
    # Using a specific model for consistency
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    exit() # Exit if the pipeline can't be loaded

# 2. Sample text
example_texts = [
    "This is a fantastic library! I love using it.",
    "The documentation could be clearer, and I encountered some bugs.",
    "It's an okay tool, neither good nor bad.",
]

# 3. Analyze the text
print("\nAnalyzing sentiments...")
try:
    results = sentiment_pipeline(example_texts)
except Exception as e:
    print(f"Error during analysis: {e}")
    results = [] # Ensure results is iterable even on error

# 4. Print the results
print("\n--- Results ---")
for i, result in enumerate(results):
    text = example_texts[i]
    label = result.get('label', 'N/A')
    score = result.get('score', 0.0)
    print(f"Text: \"{text}\"")
    print(f"Sentiment: {label}, Score: {score:.4f}\n")

print("--- Analysis Complete ---")