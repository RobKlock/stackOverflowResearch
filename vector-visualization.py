#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:38:02 2021

@author: Robert Klock
"""

import nltk
import matplotlib.pyplot as plt
import numpy as np
from csv import reader
import seaborn as sns
from numpy import linalg
import matplotlib.patheffects as PathEffects
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
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
import seaborn as sns
import string
import re

from sklearn.cluster import KMeans

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

pd.set_option('mode.chained_assignment', None)

#load data fxn
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 18))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(18):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

#load data in plain array or as a pandas dataframe
so_csv_data = load_data("training_set.csv")
train_df = pd.read_csv('training_set.csv', index_col = 0)

test_set = load_data("testing_set.csv")
test_df = pd.read_csv("testing_set.csv", index_col = 0)
# === Clean Data ====

#so_dataframe_cleaned = so_dataframe.drop(axis = 1, columns = 0)

#remove NA rows
train_df = train_df[train_df.id.notna()]
test_df = test_df[test_df.id.notna()]
#lowercase all content
#was previously so_df["content"] = so_df["content"].str.lower(), but that throws a warning
#so_dataframe_cleaned.loc[:,'content'] = so_dataframe_cleaned[:,'content'].str.lower()
#nvm
train_df["content"] = train_df["content"].str.lower()
test_df["content"] = test_df["content"].str.lower()


#some design considerations-is bag of words or straight up tokenization more useful?
#the first preserves ordering of words, the second doesnt

#remove punctuation in content

#optional - remove punc?
regex = re.compile('[%s]' % re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
                   
train_df["content"] = train_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)
test_df_cleaned = test_df["content"]
# Testing DF had to be cast to strings
test_df["content"] = test_df_cleaned.apply(lambda column : str(column))
test_df["content"] = test_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)

# Get rid of random numbers
train_df["content"] = train_df['content'].str.replace('\d+', '')
test_df["content"] = test_df['content'].str.replace('\d+', '')

# Have to make these not series
sql_code_train = train_df["content"]
sql_code_test = test_df["content"]

cleanedNoPunc = train_df
cleanedNoPunc.to_csv('CleanedNoPunc', index = False)

cleanedNoPuncTest = test_df
cleanedNoPuncTest.to_csv('CleanedNoPuncTest', index = False)

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
frames = [train_df, test_df]
all_data = pd.concat(frames)

tfidf = TfidfVectorizer(min_df = 7, max_df = .8, ngram_range = (1,3), stop_words = "english")
all_code = all_data["content"]
X_train_test = tfidf.fit_transform(all_code)
X_train_test_df = pd.DataFrame(X_train_test.toarray())
X_train_test = X_train_test_df
"""

X_train = tfidf.fit_transform(sql_code_train)
X_train_df = pd.DataFrame(X_train.toarray())
X_train = X_train_df
"""
RS = 25111993
X_train = X_train_test[0:495]
y_train = cleanedNoPunc.sql_injectable
y_train_df = y_train.to_frame()
kmeans = KMeans(n_clusters=18, random_state=0).fit(X_train)
z = pd.DataFrame(y_train.tolist()) 
digits_proj = TSNE(random_state=RS).fit_transform(X_train)
sns.palplot(np.array(sns.color_palette("hls", 2)))
scatter(digits_proj, y_train)
plt.savefig('digits_tsne-generated_18_cluster.png', dpi=120)
