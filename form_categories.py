import nltk
from nltk.stem import WordNetLemmatizer
import json
import tensorflow
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import string
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('formatted_msgs.csv', sep='\t')

tags = data.answer.unique()
answers = {}
for idx, val in enumerate(tags):
    answers[val] = idx
print('len of answers:', len(answers))

corpus = list(answers.keys())
print(corpus)

vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(corpus)
pairwise_similarity = tfidf * tfidf.T
print(pairwise_similarity.toarray())

# indices = {}
# for i in range(len(corpus)):
#     if i % 500 == 0:
#         print(i)
#     for j in range(len(corpus)):
#         if i < j and pairwise_similarity.toarray()[i][j] > 0.5:
#             if i not in indices.keys():
#                 indices[i] = []
#             indices[i].append(j)
arr = pairwise_similarity.toarray()
# np.fill_diagonal(arr, np.nan)

w = np.where(arr > 0.5)
print(len(w[0]), len(w[1]))
print(w)

categories = dict()
for i in range(len(w[0])):
    if w[0][i] < w[1][i]:
        if w[0][i] not in categories.keys():
            categories[w[0][i]] = []
        categories[w[0][i]].append(corpus[w[1][i]])

print(categories)





