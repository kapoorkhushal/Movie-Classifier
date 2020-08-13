import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords') # load english stopwords
from nltk.corpus import stopwords
from collections import Counter
from itertools import chain
import scipy.sparse as sp_sparse
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

data = pd.read_csv('movie_metadata.csv')
print("Data Shape : ",data.shape)
print("Data Sample : \n",data.head())
print("Columns present in the data: ",[i for i in data.columns])
print("Number of data points: ",data.shape[0])
name = data.movie_title
print("Duplicate Values : ",name.shape[0] - name.unique().shape[0])
label = data.genres
print("Number of Labels : ",label.unique().shape[0])
x_train,x_test,y_train,y_test = train_test_split(data['plot_keywords'],data['genres'],test_size=0.33,random_state=22)

# Dictionary of all tags from train corpus with their counts.
tags_counts = Counter(chain.from_iterable([i.split(",") for i in y_train]))

# Dictionary of all words from train corpus with their counts.
words_counts = Counter(chain.from_iterable([i.split(" ") for i in x_train]))

top_3_most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
top_3_most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"Top three most popular tags are: {','.join(tag for tag, _ in top_3_most_common_tags)}")
print(f"Top three most popular words are: {','.join(tag for tag, _ in top_3_most_common_words)}")

# We considered only the top 5,000 words, this parameter can be fine-tuned
DICT_SIZE = 5000
WORDS_TO_INDEX = {j[0]:i for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
INDEX_TO_WORDS = {i:j[0] for i,j in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    keys= [words_to_index[i] for i in text.split(" ") if i in words_to_index.keys()]
    result_vector[keys]=1
    return result_vector

x_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_train])
x_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_test])
print('X_train shape : ', x_train_mybag.shape)
print('X_test shape : ', x_test_mybag.shape)

def tfidf_features(X_train, X_test):
    """
        X_train, X_val, X_test — samples        
        return bag-of-words representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_df=0.9,min_df=5,token_pattern=r'(\S+)' )
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    return tfidf_train, tfidf_test, tfidf_vectorizer.vocabulary_

x_train_tfidf, x_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

# transform to dictionary
y_train = [set(i.split(',')) for i in y_train]
y_test = [set(i.split(',')) for i in y_test]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.fit_transform(y_test)

# For multiclass classification
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

def train_classifier(X_train, y_train, X_valid=None, y_valid=None, C=1.0, model='lr'):
    """
      X_train, y_train — training data
      
      return: trained classifier
      
    """
    
    if model=='lr':
        model = LogisticRegression(C=C, penalty='l1', dual=False, solver='liblinear',max_iter=15000)
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
    
    elif model=='svm':
        model = LinearSVC(C=C, penalty='l1', dual=False, loss='squared_hinge',max_iter=15000)
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
    
    elif model=='nbayes':
        model = MultinomialNB(alpha=1.0)
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
        
    elif model=='lda':
        model = LinearDiscriminantAnalysis(solver='svd')
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)

    return model

# Train the classifiers for different data transformations: bag-of-words and tf-idf.

# Linear NLP model using bag of words approach
classifier_mybag = train_classifier(x_train_mybag, y_train, C=1.0, model='lr')

# Linear NLP model using TF-IDF approach
classifier_tfidf = train_classifier(x_train_tfidf, y_train, C=1.0, model='lr')

y_test_predicted_labels_mybag = classifier_mybag.predict(x_test_mybag)
y_test_predicted_labels_tfidf = classifier_tfidf.predict(x_test_tfidf)
print("tfidf test predicted labels : ",y_test_predicted_labels_tfidf.shape)

"""
y_test_pred_inversed = mlb.inverse_transform(y_test_predicted_labels_tfidf)
y_test_inversed = mlb.inverse_transform(y_test)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        x_test[i],
        ','.join(y_test_inversed[i]),
        ','.join(y_test_pred_inversed[i])
    ))
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from functools import partial

def print_evaluation_scores(y_val, predicted):
    f1_score_macro = partial(f1_score,average="macro")
    f1_score_micro = partial(f1_score,average="micro")
    f1_score_weighted = partial(f1_score,average="weighted")
    
    average_precision_score_macro = partial(average_precision_score,average="macro")
    average_precision_score_micro = partial(average_precision_score,average="micro")
    average_precision_score_weighted = partial(average_precision_score,average="weighted")
    
    scores = [accuracy_score,f1_score_macro,f1_score_micro,f1_score_weighted,average_precision_score_macro,
             average_precision_score_micro,average_precision_score_weighted]
    for score in scores:
        print(score,score(y_val,predicted))

print('Bag-of-words')
print_evaluation_scores(y_test, y_test_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_test, y_test_predicted_labels_tfidf)

import matplotlib.pyplot as plt

hypers = np.arange(0.1, 1.1, 0.1)
res = []

for h in hypers:
    temp_model = train_classifier(x_train_tfidf, y_train, C=h, model='lr')
    temp_pred = f1_score(y_test, temp_model.predict(x_test_tfidf), average='weighted')
    res.append(temp_pred)

plt.figure(figsize=(7,5))
plt.plot(hypers, res, color='blue', marker='o')
plt.grid(True)
plt.xlabel('Parameter $C$')
plt.ylabel('Weighted F1 score')
plt.show()