import pandas as pd
from scipy.stats import binom
import numpy as np

def calculate_hc(human_data, ai_data):
   human_freq = pd.Series([word for sentence in human_data['human_sentence'] for word in sentence]).value_counts()
   ai_freq = pd.Series([word for sentence in ai_data['ai_sentence'] for word in sentence]).value_counts()


   # Calculate HC-discrepancy value
   n1 = sum(human_freq.values)
   n2 = sum(ai_freq.values)
   N = len(set(human_freq.index) | set(ai_freq.index))
   pi_values = []
   for w in set(human_freq.index) | set(ai_freq.index):
       x = human_freq.get(w, 0)
       nw = human_freq.get(w, 0) + ai_freq.get(w, 0)
       pw = (n1 - x) / (n1 + n2 - nw)
       pi = binom.sf(x, nw, pw)  # Use survival function to get p-value
       pi_values.append(pi)


   sorted_pi = sorted(pi_values)
   imin = next((i for i, pi in enumerate(sorted_pi) if pi >= 1 / N), None)
   i_star = np.argmax([pi for pi in sorted_pi[imin:]]) + imin
   print(i_star)
   hc_threshold = sorted_pi[i_star]
   hc_words = [w for w, pi in zip(set(human_freq.index) | set(ai_freq.index), pi_values) if pi <= hc_threshold]
   return len(hc_words), hc_words[:100]