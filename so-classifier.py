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
from scipy import stats
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
X_train = X_train_test[0:495]
y_train = cleanedNoPunc.sql_injectable
y_train_df = y_train.to_frame()

#y_train = y_train_df.to_numpy
"""
train_df = pd.DataFrame(
        y_train_df,
        columns = tfidf.get_feature_names()
        )
"""

"""
X_test = tfidf.fit_transform(sql_code_test)
X_test_df = pd.DataFrame(X_test.toarray())
X_test = X_test_df
"""
X_test = X_train_test[495:]
y_test = cleanedNoPuncTest.sql_injectable
y_test_df = y_test.to_frame()
#y_test = y_test_df.to_numpy
"""
test_df = pd.DataFrame(
        y_test_df,
        columns = tfidf.get_feature_names()
        )
"""
#compare Linear SVC and Logistic Regression

models = [LinearSVC(), LogisticRegression(random_state=3)]
#cross validation
CV= 8
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring = 'accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))


cv_df = pd.DataFrame(entries, columns = ['model_name', 'fold_idx', 'accuracy'])

print("Linear SVC ", cv_df[['accuracy']].iloc[[0,1,2,3,4,5,6,7]].mean(axis=0))
print("Logistic Regression ", cv_df[['accuracy']].iloc[[8,9,10,11,12,13,14,15]].mean(axis=0))

linearSVC = LinearSVC()
linearSVC = linearSVC.fit(X_train, y_train)
linearSVC_scores = cross_val_score(linearSVC, X_train, y_train, scoring = "neg_mean_squared_error", cv = 8)
linearSVC_rmse_scores = np.sqrt(-linearSVC_scores)

linearSVC_predictions = linearSVC.predict(X_test)
print(confusion_matrix(y_test, linearSVC_predictions))
cm_df = pd.DataFrame(confusion_matrix(y_test, linearSVC_predictions))
sn.set(font_scale=1) # for label size
ax = plt.axes()
sn.heatmap(cm_df, annot=True, annot_kws={"size": 16}, fmt="d", ax = ax) # font size
ax.set_title('Linear SVM Confusion Matrix', fontsize = 16)
plt.ylabel("Actual", fontsize = 16)
plt.xlabel("Predicted", fontsize = 16)
plt.show()

"""
# Evaluate Random Forest
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_reg_scores = cross_val_score(forest_reg, X_train, y_train, scoring = "neg_mean_squared_error", cv = 8)
forest_reg_rmse_scores = np.sqrt(- forest_reg_scores)

# Evaluate Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
tree_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_rmse = np.sqrt(tree_mse)

# Evaluate LinearSVC
linearSVC = LinearSVC()
linearSVC = linearSVC.fit(X_train, y_train)
linearSVC_scores = cross_val_score(linearSVC, X_train, y_train, scoring = "neg_mean_squared_error", cv = 8)
linearSVC_rmse_scores = np.sqrt(-linearSVC_scores)

# Evaluate Logistic Regression
logReg = LogisticRegression(random_state = 3)
logReg = logReg.fit(X_train, y_train)
train_predictions = linearSVC.predict(X_test)
lin_mse = mean_squared_error(y_test, train_predictions)
lin_rmse = np.sqrt(lin_mse)


linearSVC_predictions = cross_val_predict(linearSVC, X_test, y_test.T, cv=3)
randomForest_predicitons = cross_val_predict(forest_reg, X_test, y_test.T, cv = 4)
#print(confusion_matrix(labels, forest_pred))

#y_train_pred_log_reg = cross_val_predict(logReg, vectorized, labels, cv=3)
logisticRegression_predictions = cross_val_predict(logReg, X_test, y_test.T, cv=3)

#print(confusion_matrix(labels, y_train_pred))
print(confusion_matrix(y_test, linearSVC_predictions))
print(confusion_matrix(y_test, logisticRegression_predictions))

cm_df = pd.DataFrame(confusion_matrix(y_test, linearSVC_predictions))
sn.set(font_scale=1) # for label size
ax = plt.axes()
sn.heatmap(cm_df, annot=True, annot_kws={"size": 16}, fmt="d", ax = ax) # font size
ax.set_title('Linear SVM Confusion Matrix', fontsize = 16)
plt.ylabel("Actual", fontsize = 16)
plt.xlabel("Predicted", fontsize = 16)
plt.show()
confidence = 95


#squared_errors = (final_predictions - vectorized_test) ** 2
#np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale = stats.sem(squared_errors)))

#print(poopie.predict(so_df_c['content'][5]))





                                    
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