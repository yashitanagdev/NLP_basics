#Rule-based Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

print("Sentence 1:", sentence_1)
sentiment_score1 = TextBlob(sentence_1).sentiment.polarity
print("Sentiment Score 1:", sentiment_score1)

print("Sentence 2:", sentence_2)
sentiment_score2 = TextBlob(sentence_2).sentiment
print("Sentiment Score 2:", sentiment_score2)

print("Sentence 3:", sentence_3)
sentiment_score3 = TextBlob(sentence_3).sentiment
print("Sentiment Score 3:", sentiment_score3)

print("Sentence 4:", sentence_4)
sentiment_score4 = TextBlob(sentence_4).sentiment
print("Sentiment Score 4:", sentiment_score4)

vadar_sentiment = SentimentIntensityAnalyzer()
print(vadar_sentiment.polarity_scores(sentence_1))
print(vadar_sentiment.polarity_scores(sentence_2))
print(vadar_sentiment.polarity_scores(sentence_3))
print(vadar_sentiment.polarity_scores(sentence_4))
