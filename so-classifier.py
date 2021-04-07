#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:15:04 2020

@author: robertklock
"""

import nltk
import matplotlib.pyplot as plt
import numpy as np
from csv import reader
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
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
from tensorflow import keras
import seaborn as sn
import string
import re
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

#load data in plain array or as a pandas dataframe
so_csv_data = load_data("training_set.csv")
so_df = pd.read_csv('training_set.csv', index_col = 0)

test_set = load_data("testing_set.csv")
test_df = pd.read_csv("testing_set.csv", index_col = 0)
# === Clean Data ====

#so_dataframe_cleaned = so_dataframe.drop(axis = 1, columns = 0)

#remove NA rows
so_df_c = so_df[so_df.id.notna()]
test_nona = test_df[test_df.id.notna()]
#lowercase all content
#was previously so_df["content"] = so_df["content"].str.lower(), but that throws a warning
#so_dataframe_cleaned.loc[:,'content'] = so_dataframe_cleaned[:,'content'].str.lower()
#nvm
so_df_c["content"] = so_df_c["content"].str.lower()
test_nona["content"] = test_nona["content"].str.lower()

#some design considerations-is bag of words or straight up tokenization more useful?
#the first preserves ordering of words, the second doesnt

#remove punctuation in content

#optional - remove punc?
regex = re.compile('[%s]' % re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
so_df_c["content"] = so_df_c.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)
test_nona["content"] = test_nona.apply(lambda column : str(column), axis = 1)
test_nona["content"] = test_nona.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)
#get rid of random numbers
so_df_c["content"] = so_df_c['content'].str.replace('\d+', '')
test_nona["content"] = test_nona['content'].str.replace('\d+', '')
sql_code = so_df_c["content"]
sql_code_test = test_nona["content"]

cleanedNoPunc = so_df_c
cleanedNoPunc.to_csv('CleanedNoPunc', index = False)

cleanedNoPuncTest = test_nona
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
tfidf = TfidfVectorizer(min_df = 7, max_df = .8, ngram_range = (1,3), stop_words = "english")
tfidf2 = TfidfVectorizer(min_df = 7, max_df = .8, ngram_range = (1,3), stop_words = "english")

features = tfidf.fit_transform(sql_code)
labels = cleanedNoPunc.sql_injectable
vectorized = pd.DataFrame(
        features.todense(),
        columns = tfidf.get_feature_names()
        )
features_test = tfidf2.fit_transform(sql_code_test)
labels_test = cleanedNoPuncTest.sql_injectable

vectorized_test = pd.DataFrame(
        features_test.todense(),
        columns = tfidf2.get_feature_names()
        )
#compare Linear SVC and Logistic Regression

models = [LinearSVC(), LogisticRegression(random_state=3)]
#cross validation
CV= 8
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, vectorized, labels, scoring = 'accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))


cv_df = pd.DataFrame(entries, columns = ['model_name', 'fold_idx', 'accuracy'])

print("Linear SVC ", cv_df[['accuracy']].iloc[[0,1,2,3,4,5,6,7]].mean(axis=0))
print("Logistic Regression ", cv_df[['accuracy']].iloc[[8,9,10,11,12,13,14,15]].mean(axis=0))
forest_reg = RandomForestRegressor()
forest_reg.fit(vectorized, labels)
forest_reg_scores = cross_val_score(forest_reg, vectorized, labels, scoring = "neg_mean_squared_error", cv = 8)
forest_reg_rmse_scores = np.sqrt(- forest_reg_scores)

linearSVC = LinearSVC()
linearSVC = linearSVC.fit(vectorized, labels)

linearSVC_scores = cross_val_score(linearSVC, vectorized, labels, scoring = "neg_mean_squared_error", cv = 8)
linearSVC_rmse_scores = np.sqrt(-linearSVC_scores)
logReg = LogisticRegression(random_state = 3)
logReg = logReg.fit(vectorized, labels)
train_predictions = linearSVC.predict(vectorized)
lin_mse = mean_squared_error(labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(vectorized, labels)

tree_pred = tree_reg.predict(vectorized)
tree_mse = mean_squared_error(labels, train_predictions)
tree_rmse = np.sqrt(tree_mse)

y_train_pred = cross_val_predict(linearSVC, vectorized, labels, cv=3)
y_train_test = cross_val_predict(linearSVC, vectorized_test, labels_test.T, cv=3)

y_train_pred_log_reg = cross_val_predict(logReg, vectorized, labels, cv=3)
y_train_test_log_reg = cross_val_predict(logReg, vectorized_test, labels_test.T, cv=3)

print(confusion_matrix(labels, y_train_pred))
print(confusion_matrix(labels_test, y_train_test))
print(confusion_matrix(labels_test, y_train_test_log_reg))

cm_df = pd.DataFrame(confusion_matrix(labels_test, y_train_test))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, annot_kws={"size": 16}, fmt="d") # font size
plt.show()

#print(poopie.predict(so_df_c['content'][5]))





                                    
"""
X_train, X_test, y_train, y_test = train_test_split(cleanedNoPunc['content'], cleanedNoPunc['sql_injectable'], random_state = 0)
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["php error_reporting e_all  e_notice if  $_postregister  $getuser  $_postuser $getemail  $_postemail $getpass  $_postpass $getretypepass  $_postretypepass if$getuser  if$getemail  if$getpass if$getretypepass  if $getpass  $getretypepass   if  strlen$getemail     strstr$getmeial   strstr$getmeial  requireconfigphp $query  mysql_queryselect  from strizon where user$getuser $numrows  mysql_num_rows$query if $numrows    $query  mysql_queryselect  from strizon where email$getemail $numrows  mysql_num_rows$query if $numrows    $pass  mdmdjmsad$passjudkmal $date  datef d y $code  mdrand mysql_queryinsert into strizon values  $getuser $getpass $getemail  $code $date $query  mysql_query select  from strizon where user$getuser $numrows  mysql_num_rows$query if$numrows    $site  httplocalhostals $webmaster  strizon account activiation noreplystrizoncom $headers  from $webmaster $subject  activate your strizon account $message  thank you so much for registering with strizon please click the link below to activate your accountn $message  $siteactiveateuser$getusercode$coden $message  again thank youn $message  sincerelyn $message  the strizon team ifmail$getemail $subject $message $headers  echoyou have not successfully been registered you must activate your account from the activation link sent to b$getemailb $getuser   $getemail    else  echoan error has occured your activation email could not be sent   else echoan error has occured and your account was not created   else  echothere is already a user with that email address please choose another email address   else  echothere is already a user with that username please choose another username  mysql_close  else  echoyou must enter an valid email address to register   else  echo your passwords did not match   else  echoyou must retype your password to register   else  echoyou must enter a password to register   else  echo you must enter an email address to register   pre else  echo you must enter a username to register  pre $form  form action methodpost br input typetext nameuser value$getuser placeholderusername br input typeemail nameemail value$getemail placeholderemail address br input typepassword namepass placeholderpassword br input typepassword nameretypepass placeholderretype password br input typesubmit nameregister valueregister  echo$form "])))

#accuracy = cross_val_score(LogisticRegression(random_state = 0), features, )


#remove high and low frequency n-grams
#smaller-frequency medium n-grams are more discriminitory 

#To-do: Get more data, fix this
"""