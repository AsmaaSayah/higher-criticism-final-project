import pandas as pd
from scipy.stats import binom_test
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
# Make sure to download the necessary NLTK resources
nltk.data.path.append('./venv/lib/python3.8/site-packages/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='./venv/lib/python3.8/site-packages/nltk_data')
nltk.download('universal_tagset', download_dir='./venv/lib/python3.8/site-packages/nltk_data')
nltk.download('wordnet', download_dir='./venv/lib/python3.8/site-packages/nltk_data')

import spacy
# Load the model
nlp = spacy.load("en_core_web_sm")

# Function to filter adjectives
def filter_adjectives_spacy(words):
    doc = nlp(" ".join(words))
    return [token.text for token in doc if token.pos_ == "ADJ"]

def calculate_hc(human_data, ai_data, gamma_0=0.35):
    """
    This function identifies words that are significantly different between human-written and AI-generated text
    using the Higher Criticism (HC) method.

    Input:
    - human_data: A DataFrame containing human-written sentences with a column 'human_sentence' containing lists of words.
    - ai_data: A DataFrame containing AI-generated sentences with a column 'ai_sentence' containing lists of words.
    - gamma_0 (float): A parameter that determines up to what fraction of words should be considered for the HC calculation (default is 0.35).

    Output:
    - len(hc_words): The number of words identified as significant based on the HC threshold.
    - hc_words[:100]: A list of the first 100 words that meet the HC significance criteria.
    - more_frequent_in_ai: A list of words from hc_words that are used more frequently in AI-generated text than in human-written text.
    """
    
    # Compute word frequencies for human and AI sentences.
    human_freq = pd.Series([word for sentence in human_data['human_sentence'] for word in sentence]).value_counts()
    ai_freq = pd.Series([word for sentence in ai_data['ai_sentence'] for word in sentence]).value_counts()

    # Calculate total word counts in each set of sentences.
    n1 = human_freq.sum()
    n2 = ai_freq.sum()
    
    # Combine vocabulary from both human and AI word frequencies.
    vocabulary = set(human_freq.index) | set(ai_freq.index)
    N = len(vocabulary)
    
    # Calculate p-values using the binomial test for each word.
    pi_values = []
    for w in vocabulary:
        x = human_freq.get(w, 0)
        y = ai_freq.get(w, 0)
        nw = x + y
        pw = (n2 - y) / (n1 + n2 - nw)  # Probability under the null hypothesis.
        pi = binom_test(x, nw, pw, alternative='two-sided')
        pi_values.append((w, pi))
    
    # Sort words by p-values in ascending order.
    pi_values.sort(key=lambda x: x[1])
    sorted_words, sorted_pis = zip(*pi_values)  # Unzips into two lists
    
    # Compute the HC statistics.
    hc_scores = []
    imin = next((i for i, pi in enumerate(sorted_pis) if pi >= 1 / N), 0)
    for i in range(imin, int(gamma_0 * N)):
        z_score = np.sqrt(N) * (i / N - sorted_pis[i]) / np.sqrt((i / N) * (1 - i / N))
        hc_scores.append(z_score)
    
    # Find the maximum HC score and its index.
    i_star = imin + np.argmax(hc_scores)
    hc_threshold = sorted_pis[i_star]
    print(hc_threshold)
    
    # Select words whose p-values are below the HC threshold.
    hc_words = [w for w, pi in pi_values if pi <= hc_threshold]
    
    # Filter words that are more frequent in AI data compared to human data.
    more_frequent_in_ai = [w for w in hc_words if ai_freq.get(w, 0) > human_freq.get(w, 0)]
    
    # Return the number of discriminating words, words more frequently used by AI, and a subset of hc_words.
    return len(hc_words), hc_words[:100], more_frequent_in_ai



# Function to filter adjectives.
def filter_adjectives(word_freq):
    """
    Filters out adjectives from a word frequency dictionary.

    Parameters:
    - word_freq (dict): A dictionary where keys are words (strings) and values are their frequencies (integers).
      Example input: {'quick': 10, 'run': 5, 'beautiful': 8, 'sky': 3}
    
    Returns:
    - adjectives_only (dict): A dictionary containing only adjectives with their frequencies.
      Example output: {'quick': 10, 'beautiful': 8}
    
    What the function does:
    - Iterates over each word in the input dictionary.
    - Uses the spaCy NLP library to determine the part of speech (POS) for each word.
    - Checks if the word's POS is "ADJ" (indicating an adjective).
    - If the word is an adjective, it adds the word and its frequency to the output dictionary.
    """
    adjectives_only = {}
    for word, freq in word_freq.items():
        # Use spaCy to analyze the part of speech.
        doc = nlp(word)
        # Check if the word is an adjective.
        if doc[0].pos_ == 'ADJ':
            adjectives_only[word] = freq
    return adjectives_only