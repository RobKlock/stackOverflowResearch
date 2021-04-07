#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:52:46 2021

@author: Rob Klock
Neural net for stackoverflow research
"""

""" THIS IS A HACKY FIX USE AT YOUR OWN RISK """
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
""" end hacky fix """

from tensorflow import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
from csv import reader
# Takes in a csv and returns a list
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

'''=== Load Data === '''
# Load training and testing data as CSVs
train_csv = load_data("training_set.csv")
test_csv = load_data("testing_set.csv")

# Load training and testing data as Pandas Dataframes
train_df = pd.read_csv('training_set.csv', index_col = 0)
test_df = pd.read_csv('testing_set.csv', index_col = 0)


''' === Clean Data / Preprocessing ==== '''
# Remove NA rows
train_df = train_df[train_df.id.notna()]
train_df = train_df[train_df.id.notna()]
test_df = test_df[test_df.id.notna()]
test_df = test_df[test_df.content.notna()]

# Lowercase all content
train_df["content"] = train_df["content"].str.lower()
test_df["content"] = test_df["content"].str.lower()

''' Some design considerations-is bag of words or straight up tokenization more useful?
 the first preserves ordering of words, the second doesnt '''

# Remove punctuation in content
regex = re.compile('[%s]' % re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
train_df["content"] = train_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)
test_df["content"] = test_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)

#get rid of random numbers
train_df["content"] = train_df["content"].str.replace('\d+', '')
test_df["content"] = test_df["content"].str.replace('\d+', '')

train_sql_code = train_df["content"]
test_sql_code = test_df["content"]

tfidf = TfidfVectorizer(min_df = 8, max_df = .8, max_features=100, ngram_range = (1,3), stop_words = "english")

X_train = tfidf.fit_transform(train_sql_code)
Y_train = train_df.sql_injectable
X_train = X_train.todense()
X_train = X_train.tolist()

X_test = tfidf.fit_transform(test_sql_code)
Y_test = test_df.sql_injectable
X_test = X_test.todense()
X_test = X_test.tolist()

train_features = tfidf.get_feature_names()

train_vectorized = pd.DataFrame(
        X_train,
        columns = tfidf.get_feature_names()
        )

test_vectorized = pd.DataFrame(
        X_test,
        columns = tfidf.get_feature_names()
)

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
Y_test = Y_test.to_numpy()
Y_train = Y_train.to_numpy()

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation = "sigmoid"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy",
              optimizer="sgd",
              metrics=["binary_accuracy"])
print(model.summary)
history = model.fit(X_train, Y_train, epochs = 200,
                    validation_data=(X_test, Y_test), verbose = 1)
"""
#imbd = keras.datasets.imdb
#(train_data, train_labels), (test_data, test_labels) = imbd.load_data(num_words = 10000)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(446, 10))
model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.InputLayer(input_shape = (495,100)))
model.add(keras.layers.Dense(600, activation = "relu"))
model.add(keras.layers.Dense(50, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, Y_train, epochs = 20,
                    validation_data=(X_test, Y_test), verbose = 1)
"""
"""
test_vectorized = pd.DataFrame(
        X_test.todense(),
        columns = tfidf.get_feature_names()
        )
"""