import sklearn
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import re

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_features(text_list):
    return text_list
    #return [clean(t) for t in text_list]


class Model():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            stop_words='english',
            sublinear_tf=True)
        #self.classifier = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=1000)
        #self.classifier = LinearSVC(max_iter=1000)
        self.classifier = ComplementNB(alpha=0.5, fit_prior=True)
        #self.classifier = RidgeClassifier()
        #self.classifier = SGDClassifier(loss='log', max_iter=1000, random_state=42)
        #self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X_list, y_list):
        X_vect = self.vectorizer.fit_transform(X_list)
        self.classifier.fit(X_vect, y_list)

    def predict(self, X_list):
        X_vect = self.vectorizer.transform(X_list)
        return self.classifier.predict(X_vect).tolist()



# REPLACE TO YOUR FEATURE EXTRACTOR
def extract_features1(text_list):
    feature_list = []
    for text in text_list:
        words = text.lower().split()
        feature = words[0]
        feature_list.append(feature)
    return feature_list


# REPLACE TO YOUR MODEL
class Model1():
    def __init__(self):
        self.vocab = dict()

    def fit(self, X_list, y_list):
        for X, y in zip(X_list, y_list):
            self.vocab[X] = y
        self.default_category = y

    def predict_sample(self, X):
        if X in self.vocab:
            return self.vocab[X]
        return self.default_category

    def predict(self, X_list):
        answer = []
        for X in X_list:
            answer.append(self.predict_sample(X))
        return answer


def read_data_known(file_path):
    df = pd.read_csv(file_path, sep='\t')

    df = df[df['category'] != 'UNKNOWN']

    N = len(df)
    df_train = df.iloc[:N // 2]
    df_test = df.iloc[N // 2:]

    y_train = df_train['category'].str.strip().tolist()
    y_test = df_test['category'].str.strip().tolist()

    X_train = extract_features(df_train['text'].tolist())
    X_test = extract_features(df_test['text'].tolist())
    return X_train, y_train, X_test, y_test


def read_data(file_path):
    df = pd.read_csv(file_path, sep='\t')

    df_train = df[df['category'] != 'UNKNOWN']
    df_test = df[df['category'] == 'UNKNOWN']

    y_train = df_train['category'].str.strip().tolist()

    X_train = extract_features(df_train['text'].tolist())
    X_test = extract_features(df_test['text'].tolist())
    return X_train, y_train, X_test



MODE = "LOCAL" # "SUBMIT" OR "LOCAL"


# READ DATA
if MODE == "LOCAL":
    X_train, y_train, X_test, y_test = read_data_known("input.txt")
else:
    X_train, y_train, X_test = read_data("input.txt")
    y_test = None


# TRAIN MODEL
model = Model()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# OPTIONAL METRICS ONLY FOR LOCAL TEST
if y_test:
    ok = (np.array(y_test) == np.array(y_predict)).sum()
    accuracy = ok / len(y_test) * 100
    print("ACCURACY:", accuracy)


# WRITE ANSWER
with open("output.txt", "w") as f:
    f.write("\n".join(y_predict) + '\n')
