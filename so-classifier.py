#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:15:04 2020

@author: robertklock
"""

import matplotlib.pyplot as plt
from csv import reader
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import re
pd.set_option('mode.chained_assignment', None)

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
# Lowercase all content
train_df["content"] = train_df["content"].str.lower()

''' Some design considerations-is bag of words or straight up tokenization more useful?
 the first preserves ordering of words, the second doesnt '''

# Remove punctuation in content
regex = re.compile('[%s]' % re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
train_df["content"] = train_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)
test_df["content"] = test_df.apply(lambda column: re.sub(regex, '', column['content']), axis = 1)

#get rid of random numbers
train_df["content"] = train_df["content"].str.replace('\d+', '')
test_df["content"] = test_df["content"].str.replace
sql_code = train_df["content"]

'''
 Old, for giving the clean data elsewhere
cleanedNoPunc = train_df
cleanedNoPunc.to_csv('CleanedNoPunc', index = False)
'''


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
tfidf = TfidfVectorizer(min_df = 8, max_df = .8, ngram_range = (1,6), stop_words = "english")

features = tfidf.fit_transform(sql_code)
labels = train_df.sql_injectable

vectorized = pd.DataFrame(
        features.todense(),
        columns = tfidf.get_feature_names()
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

linearSVC = LinearSVC()
linearSVC = linearSVC.fit(vectorized, labels)

y_train_pred = cross_val_predict(linearSVC, vectorized, labels, cv=3)
print(confusion_matrix(labels, y_train_pred))

# Plot non-normalized confusion matrix
"""
confusionMatrix = plot_confusion_matrix(linearSVC, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
disp.ax_.set_title("Confusion matrix, without normalization")

print(title)
print(disp.confusion_matrix)
"""

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