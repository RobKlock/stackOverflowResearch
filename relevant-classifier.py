#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:44:34 2021

@author: Rob Klock 
    Relevant Classifier
"""

import nltk
import matplotlib.pyplot as plt
import numpy as np
from csv import reader
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import tensorflow as tf
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tensorflow import keras
import seaborn as sn
import string
import re
pd.set_option('mode.chained_assignment', None)

def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

dataset = pd.read_csv('to_use.csv', index_col = 0)
dataset.drop_duplicates(subset="code", keep=False, inplace=True)

#remove NA rows
dataset = dataset[dataset.code.notna()]
#lowercase all content
#was previously so_df["content"] = so_df["content"].str.lower(), but that throws a warning
#so_dataframe_cleaned.loc[:,'content'] = so_dataframe_cleaned[:,'content'].str.lower()
#nvm
dataset["code"] = dataset["code"].str.lower()

#remove punctuation in code
regex = re.compile('[%s]' % re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
dataset["code"] = dataset.apply(lambda column: re.sub(regex, '', column['code']), axis = 1)

y_train = dataset["label"]
y_train = y_train.to_frame()
for i in range (0, y_train.size): 
    if y_train[i]['label'] == "good":
        y_train[i]['label'] = 1
    else:
        y_train[i]['label'] = 0

# Get rid of random numbers
dataset["code"] = dataset['code'].str.replace('\d+', '')

# Have to make these not series
sql_code = dataset["code"]

#query = re.sub(regex, ' ', query)

#tokenize
#we use TF-IDF value instead of counter, which measures how important a term is in a document
#specifically measures how often a term appears in a document, inversely related to how often it appears in all documents
#so_df_c["content"] = so_df_c.apply(lambda column: nltk.word_tokenize(column['content']), axis = 1)       
#future design could add more rules to a treebank tokenizer?
#how can we normalize this text?
#do we want to remove stopwords? how best to define those


#vectorize the code so we have a numerical representtion of sql queries
#Each code content is replaced by a huge vector of numbers 
#we use grams of 1, 2, and 3 but may need to increase
#we used max and min df (document frequency) to get rid of stopwords (will need to be adjusted)
#then, normalized everything so no value exceeds one (or goes below zero)

tfidf = TfidfVectorizer(min_df = 7, max_df = .8, ngram_range = (1,3), stop_words = "english")
all_code = dataset["code"]
X_train = tfidf.fit_transform(dataset["code"])

# Evaluate Random Forest
forest_reg = RandomForestClassifier()
forest_reg.fit(X_train, y_train)
forest_reg_scores = cross_val_score(forest_reg, X_train, y_train, scoring = "neg_mean_squared_error", cv = 8)
forest_reg_rmse_scores = np.sqrt(- forest_reg_scores)
forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring = "accuracy", cv = 8)
print(forest_scores.mean())
forest_scores_test = cross_val_score(forest_reg, X_test, y_test, scoring = "accuracy", cv = 8)
print("forest test", forest_scores_test.mean())
