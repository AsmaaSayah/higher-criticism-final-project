import pandas as pd
from scipy.stats import binom_test
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
# Make sure to download the necessary NLTK resources
nltk.data.path.append('./venv/lib/python3.8/site-packages/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='./venv/lib/python3.8/site-packages/nltk_data')
nltk.download('wordnet', download_dir='./venv/lib/python3.8/site-packages/nltk_data')

def calculate_hc(human_data, ai_data, gamma_0=0.35):
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
        pi_values.append(pi)
    
    # Sort the p-values in ascending order.
    sorted_pi = np.sort(pi_values)
    
    # Compute the HC statistics.
    hc_scores = []
    imin = next((i for i, pi in enumerate(sorted_pi) if pi >= 1 / N), 0)
    for i in range(imin, int(gamma_0 * N)):
        z_score = np.sqrt(N) * (i / N - sorted_pi[i]) / np.sqrt((i / N) * (1 - i / N))
        hc_scores.append(z_score)
    
    # Find the maximum HC score and its index.
    i_star = imin + np.argmax(hc_scores)
    hc_threshold = sorted_pi[i_star]
    print(hc_threshold)
    # Select words whose p-values are below the HC threshold.
    hc_words = [w for w, pi in zip(vocabulary, pi_values) if pi <= hc_threshold]
     # Filter words that are more frequent in AI data compared to human data.
    more_frequent_in_ai = [w for w in hc_words if ai_freq.get(w, 0) > human_freq.get(w, 0)]
    
    # Return the number of discriminating words, words more frequently used by AI, and a subset of hc_words.
    return len(hc_words), hc_words[:100], more_frequent_in_ai



def filter_adjectives(word_list):
    # POS tag the words
    pos_tagged = nltk.pos_tag(word_list)
    # Keep only adjectives ('JJ', 'JJR', 'JJS' are tags for adjectives)
    adjectives = [word for word, tag in pos_tagged if tag in ('JJ', 'JJR', 'JJS')]
    
    return adjectives
