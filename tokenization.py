from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, TreebankWordDetokenizer

corpus = "My name is Yashita. Hello welcome! Where are you? It's mine."

sentences = sent_tokenize(corpus)

words = word_tokenize(corpus)

words1 = word_tokenize(sentences[0])

words2 = wordpunct_tokenize(corpus)

tokenizer = TreebankWordDetokenizer()
print(tokenizer.tokenize(words1))
detokenized = tokenizer.detokenize(words1)
print(detokenized)