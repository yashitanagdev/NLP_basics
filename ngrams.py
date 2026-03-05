import nltk
import pandas as pd
import matplotlib.pyplot as plt

tokens = ['the', 'life', 'of', 'civilization','the', 'life', 'of', 'love']
print(tokens)
unigrams = pd.Series(nltk.ngrams(tokens, 1)).value_counts()
print(unigrams)
unigrams.sort_values().plot(kind='barh', color='skyblue')
plt.title('Unigram Frequency')
plt.show()

bigrams = pd.Series(nltk.ngrams(tokens, 2)).value_counts()
print(bigrams)
bigrams.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Bigram Frequency')