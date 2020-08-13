import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('movie_metadata.csv')
print(df.shape)
print(df.head())

label = df.genres1
print(type(label))
print(label.head())
print(label.unique())

x_train,x_test,y_train,y_test = train_test_split(df['plot_keywords'],label,test_size=0.33,random_state=22)
Tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

tfidf_train = Tfidf_vectorizer.fit_transform(x_train)
tfidf_test = Tfidf_vectorizer.transform(x_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train,y_train)

pred = nb_classifier.predict(tfidf_test)
score = metrics.accuracy_score(y_test,pred)
print(f'Accuracy: {round(score*100,2)}%')

matrix = metrics.confusion_matrix(y_test,pred,labels=label.unique())
print(matrix)
plt.matshow(matrix)
plt.show()