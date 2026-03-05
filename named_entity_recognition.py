import nltk

sentence="The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel, whose company specialized in building metal frameworks and structures."
words = nltk.word_tokenize(sentence)
tagg_var = nltk.pos_tag(words)

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
print(nltk.ne_chunk(tagg_var).draw())