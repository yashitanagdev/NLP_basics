from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("fairly", pos="v"))
print(lemmatizer.lemmatize("better", pos="a"))