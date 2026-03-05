# lDA and LSA
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora
from gensim.models import LsiModel


data = pd.read_csv('news_articles.csv')
# print(data.head())

articles = data['content'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
articles = articles.apply(word_tokenize)
articles = articles.apply(lambda x: [word for word in x if word not in stopwords.words('english')])
stemmer = PorterStemmer()
articles = articles.apply(lambda x: [stemmer.stem(word) for word in x])

# print(articles)

dictionary = corpora.Dictionary(articles)
# print(dictionary)

doc_term = [dictionary.doc2bow(article) for article in articles]
# print(doc_term)

num_topics = 2
lda_model = gensim.models.LdaModel(corpus=doc_term, id2word=dictionary, num_topics=num_topics)
# print(lda_model.print_topics(num_words=5))

lsi_model = LsiModel(corpus=doc_term, id2word=dictionary, num_topics=num_topics)
print(lsi_model.print_topics(num_words=5))