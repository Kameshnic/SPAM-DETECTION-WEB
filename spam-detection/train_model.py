import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("D:/Machine-Learning/spam-detection/spam.csv", encoding='latin1')


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})


df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})


X = df['Message']
y = df['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_features = tfidf_vectorizer.fit_transform(X_train)
X_test_features = tfidf_vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_features, y_train)


train_accuracy = accuracy_score(y_train, model.predict(X_train_features))
test_accuracy = accuracy_score(y_test, model.predict(X_test_features))

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

joblib.dump(model, 'D:/Machine-Learning/spam-detection/spam_model.pkl')
joblib.dump(tfidf_vectorizer, 'D:/Machine-Learning/spam-detection/tfidf_vectorizer.pkl')
