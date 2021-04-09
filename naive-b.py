from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd

# Load in dataset
df = pd.read_csv('to_use.csv')

# Remove column names. R included them for some reason.
df = df.drop(0)

# Map the labels to integer values. secure=0, sql_injectable=1
df['label'] = df.label.map({'good': 0, 'bad': 1})


# nltk.download()
# Tokenize the code into single words
df['code'] = df['code'].apply(nltk.word_tokenize)


stemmer = PorterStemmer()

df['code'] = df['code'].apply(lambda x: [stemmer.stem(y) for y in x])


# This converts the list of words into space-separated strings
df['code'] = df['code'].apply(lambda x: ' '.join(x))

# Transform the data into occurrences, which is what we feed into our model
count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['code'])

# We could leave it as the simple word-count per message, but it is better to use Term Frequency Inverse Document Frequency, more known as tf-idf:
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(
    counts, df['label'], test_size=0.8, random_state=69)

model = MultinomialNB().fit(X_train, y_train)

# Evaluate the model
predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

# Check error balance
print(confusion_matrix(y_test, predicted))
