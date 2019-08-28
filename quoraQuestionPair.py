#!/usr/bin/env python
# coding: utf-8

# It takes more than 12 hours to run this whole code.

import numpy as np
import pandas as pd

import scipy # for scientific computing and technical computing. This library depends on numpy library.

import re
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
    
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor


"""
Source: https://www.kaggle.com/c/quora-question-pairs
Data fields

id -                   the id of a training set question pair
qid1, qid2 -           unique ids of each question (only available in train.csv)
question1, question2 - the full text of each question
is_duplicate -         the target variable, set to 1 if question1 and question2 have essentially 
                       the same meaning, and 0 otherwise.
"""

df = pd.read_csv('../../kaggle/train.csv') # read data
df.head(10)

df.columns

# Find number of rows and columns
df.shape # 404290 rows, 6 columns

# Give basic statistics for columns that have numerical values
df.describe()


df.info()


"""Check for missing values. Python by default replaces 
missing values by NaN."""
df.isnull().sum()
# Found 1 NaN value in question1.
# Found 2 NaN value in question2.


df.shape


""" Removing NaN values """
df=df.dropna() # drop NaN values 
df.shape # there are now 404287 entries with no NaN values.
# dropped 1 record from question1 and 2 records from question2; 
# hence overall 3 records dropped.


df.isnull().sum() # Now there are no NaN values.


nume = df.is_duplicate.sum() # give me sum of is_duplicate==1
#nume # we have 149263 duplicate questions given by default

denom = df["is_duplicate"].count() # give me total of is_duplicate.
#denom   # 404287

fraction_of_duplicate_to_total = nume/denom
fraction_of_duplicate_to_total  
# We have 36.92% of duplicate questions by default
# Any number we get more than 37% in our work will be an achievement.


# df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True) # drop the line with heading

# Print 10 Question pairs (horizontal pairs like in df.head(10) above)
a = 0
for i in range(a,a+10):
    print(df.question1[i]) # print first ith question
    print(df.question2[i]) # print 2nd ith question
    print()                # print empty line


# A plot to see how many is_duplicate is really duplicate (=1) and not really duplicate (=0)
# Approximately 250k (with a tag=0) questions are NOT really duplicate, 
# and only about 150k (with a tag=1) seem to be duplicate. X-axis: Label distribution in data 

print("No of data points: ",df.shape[0])
print(df["is_duplicate"].value_counts()) # show 255024 0's and 149263 1's out of 404287 total is_duplicate counts.

is_dup = df['is_duplicate'].value_counts()
color = sns.color_palette()
plt.figure(figsize=(8,4))

sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Duplicate', fontsize=12)

plt.show()



# Clean up the text in questions.
# https://towardsdatascience.com/finding-similar-quora-questions-with-bow-tfidf-and-random-forest-c54ad88d1370
# This list is so extensive that we can look back at it as a reference when 
# needed for other regular expression related work.


SPECIAL_TOKENS = {'non-ascii': 'non_ascii_word'}

def clean(text, stem_words=True):
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

    # Empty question
    
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE) # replace whats by what is and ignore case
    text = re.sub("\'ve", " have ", text) # replace 've by have
    text = re.sub("can't", "can not", text) # replace can't by can not
    text = re.sub("n't", " not ", text) # replace n't by not
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE) # replace i'm by i am and ignore case
    text = re.sub("\'re", " are ", text) # replace 're by are
    text = re.sub("\'d", " would ", text) # replace 'd by would
    text = re.sub("\'ll", " will ", text) # replace 'll by will
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE) # replace e.g. by eg and ignore case
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE) # replace b.g. by bg and ignore case
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text) 
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    # add padding to punctuations and special chars, we still need them later
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) 
    
    # indian dollar
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # replace the float numbers with a random number, it will be parsed as number afterward, 
    # and also been replaced with word "number"
    
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
  
    
    # Try 1. keeping and 2. removing puncutation to see if the results change.
    # Result: Tried with random forest, there was no change in the result (except a slight change in confusion matrix).
    # Remove punctuation from text.
    text = ''.join([c for c in text if c not in punctuation]).lower()
       
    # Return a list of words
    return text
    
df['question1'] = df['question1'].apply(clean)
df['question2'] = df['question2'].apply(clean)


# Look at clean data
df.head(10)


# After cleaning the text, we preview those question pairs again.
a = 0 
for i in range(a,a+10):
    print(df.question1[i])
    print(df.question2[i])
    print()


# Use word level CountVectorizer to identify whether qestion1 is duplicate to question2 or not.

# CountVectorizer documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# Use word level CountVectorizer to identify whether qestion1 
# is duplicate to question2 or not.

# r'\w{1,}' means 1 or more words
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}') 

q1_trans = count_vect.fit_transform(df['question1'].values)
q2_trans = count_vect.fit_transform(df['question2'].values)
labels = df['is_duplicate'].values

# Stack two questions using scipy sparse matrix, horizontally (column wise), 
# so as to make X variables as a 2-dimensional matrix, wehre
# we can think of like we have two variables (features): q1_trans, and 
# q2_trans. Here we are not mixing two questions (q1_trans, and q2_trans) 
# to make one bag of words, but we are stacking them side by side 
# (still as two elements in a row of the matrix) to make X as a 2D matrix.

X = scipy.sparse.hstack((q1_trans,q2_trans)) 
y = labels
print(X.shape)
print(y.shape)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#
# Using random forest classifier
#

from sklearn.ensemble import RandomForestClassifier
# build a pipeline using RandomForestClassifier()
rf_clf = Pipeline([('tfidf', TfidfTransformer()),('RF', RandomForestClassifier()),])

# Feed the data through the pipeline
rf_clf.fit(X_train, y_train)   


# In[19]:


# Form a prediction set using random forest classifier
predictionRF = rf_clf.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictionRF))
print("\nOverall Accuracy:\n    {:.2f}%".format(accuracy_score(y_test, predictionRF) * 100))
print("\nClassification report:\n", metrics.classification_report(y_test,predictionRF))

# Confusion Matrix:
#  [[70289  6320]
#  [20682 23996]]

# Overall Accuracy:
#     77.74%

# Classification report:
#                precision    recall  f1-score   support

#            0       0.77      0.92      0.84     76609
#            1       0.79      0.54      0.64     44678

#    micro avg       0.78      0.78      0.78    121287
#    macro avg       0.78      0.73      0.74    121287
# weighted avg       0.78      0.78      0.77    121287

# The accuracy here is 77.74% which is rather low but far better than the default value of 37%.

#
# Build a pipeline using linear support vector classifier.
#

# build a pipeline using LinearSVC()
# svc_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('svc', LinearSVC()),])
# Do not use CountVectorizer as explained in here:
# https://stackoverflow.com/questions/50192763/python-sklearn-pipiline-fit-attributeerror-lower-not-found?rq=1
"""
Either remove step ('vect', CountVectorizer()) or use TfidfTransformer instead of TfidfVectorizer 
as TfidfVectorizer expects array of strings as an input and CountVectorizer() returns a matrix of 
occurances (i.e. numeric matrix).

Per default TfidfVectorizer(..., lowercase=True) will try to "lowercase" all strings, hence 
the “AttributeError: lower not found” error message.
"""
# build a pipeline using LinearSVC()
svc_clf = Pipeline([('tfidf', TfidfTransformer()),('svc', LinearSVC()),])

# Feed the data through the pipeline
svc_clf.fit(X_train, y_train)   


# Form a prediction set using linear support vector classifier.
predictionSV = svc_clf.predict(X_test)

# Get the score report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictionSV))
print("\nOverall Accuracy:\n    {:.2f}%".format(accuracy_score(y_test, predictionSV) * 100))
print("\nClassification report:\n", metrics.classification_report(y_test,predictionSV))


# Confusion Matrix:
#  [[64560 12049]
#  [17106 27572]]

# Overall Accuracy:
#     75.96%

# Classification report:
#                precision    recall  f1-score   support

#            0       0.79      0.84      0.82     76609
#            1       0.70      0.62      0.65     44678

#    micro avg       0.76      0.76      0.76    121287
#    macro avg       0.74      0.73      0.73    121287
# weighted avg       0.76      0.76      0.76    121287


# The accuracy we got is 75.96% with this method. We try another method next to improve it if possible.



# Use Character Level TF-IDF to identify whether qestion1 is duplicate to question2 or not.

# Note that TF-IDF based method does not perform well when there is no vocabulary overlap 
# but there is semantic similarity between sentences. 
# So we must use word2vec (toward the end section of this project). 

# r'\w{1,}' means 1 or more words
# n-gram means sequence of n words. 
# ngram_range=(2,3) means we use bigrams or trigrams.
# max_features=5000 means we use 5000 columns at maximum.

tfidf_vect_ngram_chars=TfidfVectorizer(analyzer='char',token_pattern=r'\w{1,}',ngram_range=(2,3),max_features=5000)

q1_trans = tfidf_vect_ngram_chars.fit_transform(df['question1'].values)
q2_trans = tfidf_vect_ngram_chars.fit_transform(df['question2'].values)
labels = df['is_duplicate'].values

# Stack two questions, using sparse matrix, horizontally (column wise) 
# so as to make X variable a 2-dimensional matrix, wehre
# we can think of like we have two variables (features): q1_trans, and q2_trans.
# Here we are not mixing two questions (q1_trans, and q2_trans) to make one bag of words, 
# but we are stacking them side by side (still as two elements in a row of the matrix) to make X as a 2D matrix.
X = scipy.sparse.hstack((q1_trans,q2_trans))
y = labels



q1_trans.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



#
# build a pipeline using LinearSVC()
#

svc_clfChar = Pipeline([('tfidf', TfidfTransformer()),('svc', LinearSVC()),])

# Feed the data through the pipeline
svc_clfChar.fit(X_train, y_train)   



# Form a prediction set
predictionSVchar = svc_clfChar.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictionSVchar))
print("\nOverall Accuracy:\n    {:.2f}%".format(accuracy_score(y_test, predictionSVchar) * 100))
print("\nClassification report:\n", metrics.classification_report(y_test,predictionSVchar))


# Confusion Matrix:
#  [[65707 10902]
#  [20255 24423]]

# Overall Accuracy:
#     74.31%

# Classification report:
#                precision    recall  f1-score   support

#            0       0.76      0.86      0.81     76609
#            1       0.69      0.55      0.61     44678

#    micro avg       0.74      0.74      0.74    121287
#    macro avg       0.73      0.70      0.71    121287
# weighted avg       0.74      0.74      0.74    121287

# The accuracy we got now went down from 75% to 74% with this method - not good.
# We add Xgboost next to improve accuracy.




# Use Character Level TF-IDF + Xgboost to identify whether qestion1 is duplicate to question2 or not.

# Character Level TF-IDF + Xgboost (Extreme Gradient Boosting)

# XGBoost (Extreme Gradient Boosting):  https://www.datacamp.com/community/tutorials/xgboost-in-python
# XGBoost is an implementation of gradient boosted decision trees designed for speed 
# and performance that is dominative competitive machine learning. It has shown better performance 
# on a variety of machine learning benchmark datasets. XGBoost internally has parameters for 
# cross-validation, regularization, user-defined objective functions, missing values, 
# tree parameters, scikit-learn compatible API etc.

# What is boosting? https://www.datacamp.com/community/tutorials/xgboost-in-python
# Boosting is a sequential technique which works on the principle of an ensemble. 
# It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, 
# the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted 
# correctly are given a lower weight and the ones miss-classified are weighted higher. Note that a 
# weak learner is one which is slightly better than random guessing. For example, a decision tree whose 
# predictions are slightly better than 50%.


# TfidfVectorizer -- Brief Tutorial:  tf: term frequency, idf: inverse document frequency
#https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

# The goal of using tf-idf is to scale down the impact of tokens that occur very frequently 
# in a given corpus and that are hence empirically less informative than features that occur 
# in a small fraction of the training corpus. We want low positive weights for frequent terms 
# and high weights for rare terms.

# TF: How often does the term appear in a document? The more often, the higher the weight.
# IDF: How often does the term appear in all documents in the collection? The more often, the lower the weight.
# The product TF * IDF will scale down the impact of tokens that occur very frequently, but are
# less informative, in the document.

# TF-IDF Source - see last example here: https://en.wikipedia.org/wiki/Tf–idf 

import datetime
st = datetime.datetime.now() # current time

xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, 
                              gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, 
                              subsample=0.8).fit(X_train, y_train) 
xgb_prediction = xgb_model.predict(X_test)

print('character level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
print('character level tf-idf test score:', f1_score(y_test, xgb_model.predict(X_test), average='macro'))


print("\nConfusion Matrix:\n", confusion_matrix(y_test, xgb_prediction))
print("\nOverall Accuracy:\n    {:.2f}%".format(accuracy_score(y_test, xgb_prediction) * 100))
print("\nclassification_report\n", classification_report(y_test, xgb_prediction))

nd = datetime.datetime.now() # end time now
print("Code run-time: ", nd-st) # it's the total time to run the code (finish time -start time)


# character level tf-idf training score: 0.997985723769572
# character level tf-idf test score: 0.8048011235004698

# Confusion Matrix:
#  [[69878  6731]
#  [14418 30260]]

# Overall Accuracy:
#     82.56%

# classification_report
#                precision    recall  f1-score   support

#            0       0.83      0.91      0.87     76609
#            1       0.82      0.68      0.74     44678

#    micro avg       0.83      0.83      0.83    121287
#    macro avg       0.82      0.79      0.80    121287
# weighted avg       0.82      0.83      0.82    121287

# Code run-time:  2:36:07.556734


# Now the accuracy is 82.56% which is the best we have.



#
# All code below this runs independently of any code written above.
#




#
# To capture sentiment, we use MaLSTM neural net.
# https://github.com/amanraj209/siamese-lstm-sentence-similarity
#

import re
import gensim
import logging
import itertools

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Clean up the text
def text_to_word_list(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()

    return text

# We use the 300-dimensional word2vec embeddings from google which can capture intricate
# inter-word relationships such as vec(king) − vec(man) + vec(woman) ≈ vec(queen).
# This is publicly available at: code.google.com/p/word2vec
def make_w2v_embeddings(df, embedding_dim=300):
    vocabs = {}
    vocabs_cnt = 0
    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    stops = set(stopwords.words('english'))

    word2vec = KeyedVectors.load_word2vec_format("/Users/rameshsubedi/Downloads/nlpDownload/GoogleNews-vectors-negative300.bin", binary=True)

    for index, row in df.iterrows():
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        for question in ['question1', 'question2']:

            q2n = []
            for word in text_to_word_list(row[question]):
                if word in stops:
                    continue
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)
    embeddings[0] = 0

    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):
    X = {'q1': df['question1_n'], 'q2': df['question2_n']}

    for dataset, side in itertools.product([X], ['q1', 'q2']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


class ManhattanDistance(Layer):
    def __init__(self, **kwargs):
        super(ManhattanDistance, self).__init__(**kwargs)
        self.result = None

    def build(self, input_shape):
        super(ManhattanDistance, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    



import os
import matplotlib

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Input, Embedding, LSTM

matplotlib.use('Agg')

#ROOT_PATH = os.path.abspath('/Users/rameshsubedi/Downloads/kaggle')
ROOT_PATH = os.path.abspath('') # It's /Users/rameshsubedi
TRAIN_CSV = 'train.csv'

gpus = 2
#batch_size = 256 * gpus
batch_size = 128 * gpus

n_epoch = 50 # we will be training the whole data (the training data part) 50 times.
#  To learn more about epoch, batch, look here: 
# https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks
"""
An EPOCH describes the number of times the algorithm sees the entire data set. 
So, each time the algorithm has seen all samples in the dataset, an epoch has completed.

An ITERATION describes the number of times a BATCH of data passed through the algorithm. 
In the case of neural networks, that means (or ITERATION means) the forward pass and backward pass. 
So, every time you pass a batch of data through the NN, you completed an iteration.

Example: if you have 1000 training examples (rows), and your batch size is 500, 
then it will take 2 iterations to complete 1 epoch.

In this analysis there are 404287 rows, and batch size is 256, so it will take 404287/256=1579.246
iterations complete one epoch.

"""
#n_hidden = 256
n_hidden = 128

# using 300 dimension for embedding i.e. there will be 300 vectors for each word in the corpora 
# represented for neural network model.
embedding_dim = 300
max_seq_length = 25  # length of the longest question has 25 words



def prepare_data():
    train_df = pd.read_csv(os.path.join(ROOT_PATH, TRAIN_CSV))
    for q in ['question1', 'question2']:
        train_df[q + '_n'] = train_df[q]

    train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim)
    #validation_size = int(len(train_df) * 0.1)
    validation_size = int(len(train_df) * 0.3) # 30% for test (here validation means test), 70% for train

    X = train_df[['question1_n', 'question2_n']]
    Y = train_df['is_duplicate']

    #X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=42)


    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    assert X_train['q1'].shape == X_train['q2'].shape
    assert len(X_train['q1']) == len(Y_train)

    return X_train, X_validation, Y_train, Y_validation, embeddings




def prepare_model(embeddings):
    shared_model = Sequential() 
    # Set trainble=False so that the gradient decent will not optimize the embeddings.
    shared_model.add(Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
    shared_model.add(LSTM(n_hidden))

    q1_input = Input(shape=(max_seq_length,), dtype='int32')
    q2_input = Input(shape=(max_seq_length,), dtype='int32')

    malstm_distance = ManhattanDistance()([shared_model(q1_input), shared_model(q2_input)])
    
    # The Model below is due from:  from tensorflow.python.keras.models import Model
    model = Model(inputs=[q1_input, q2_input], outputs=[malstm_distance]) 

    # if gpus >= 2:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
   
   # Adam() from here:   from tensorflow.python.keras.optimizers import Adam
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()

    return model



def train_model(X_train, X_validation, Y_train, Y_validation, model):
    training_start_time = time()

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(ROOT_PATH, 'model/weights.{epoch:02d}.h5'),
        verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', period=1)
        
    malstm_trained = model.fit(
        [X_train['q1'], X_train['q2']], Y_train, batch_size=batch_size, epochs=n_epoch,
        validation_data=([X_validation['q1'], X_validation['q2']], Y_validation), 
        callbacks=[checkpointer])
    
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    model.save(os.path.join(ROOT_PATH, 'model/siamese-lstm-weights.h5')) # This file will be created when we run the main function below.

    return malstm_trained



def plot_accuracy_and_loss(malstm_trained):
    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left') 

    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right') 

    plt.tight_layout(h_pad=1.0)
    plt.savefig(os.path.join(ROOT_PATH, 'model/history-graph.png')) # This graph will be made when we run the main function below.
    print(str(malstm_trained.history['val_acc'][-1])[:6] + "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")



if __name__ == '__main__':
    X_train, X_validation, Y_train, Y_validation, embeddings = prepare_data()
    model = prepare_model(embeddings)
    malstm_trained = train_model(X_train, X_validation, Y_train, Y_validation, model) 
    plot_accuracy_and_loss(malstm_trained)
    



# Conclusion:

# 1. Based on Character Level TF-IDF + Xgboost (which does not capture the sentiments of question-pairs) 
#    we get the best 'recall' of 68%. That is, the proportion of duplicate questions that our model is 
#    able to detect over the total amount of duplicate questions is 0.68.
#    The maximum accuracy obtined is 82.56%

# 2. To capture the sentiment of question-pairs, we use word2vec embeddings and employ 
#    MaLSTM (Manhattan Long Short-Term Memory) neural network for the analysis.
#    We find question-pair's duplicate accuracy of 82.97%.

# 3. The accuracies obtained from both methods are almost the same.

