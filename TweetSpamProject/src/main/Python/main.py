import os
from tkinter import Image

import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import sklearn

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import *
from sklearn.naive_bayes import BernoulliNB
#from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#from sklearn.tree import export_graphviz
from six import StringIO
#import pydotplus
import sys

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
plt.close('all')


stemmer = PorterStemmer()


def hashtags(message):
    hash = re.findall("#(\\w+)", message)
    return hash


def removeHashtags(message):
    return re.sub("(\\s*#(\\w+)\\s*)+", " ", message)


def mentions(message):
    ment = re.findall("@(\\w+)", message)
    return ment


def removeMentions(message):
    return re.sub("(\\s*@(\\w+)\\s*)+", " ", message)


def urls(message):
    url = re.findall('https?://((?:[-\\w.]|(%[\\da-fA-F]{2}))/?)+(#\\w+(=\\w+)*)*', message)
    return url


def removeUrls(message):
    return re.sub('(\\s*https?://((?:[-\\w.]|(%[\\da-fA-F]{2}))/?)+(#\\w+(=\\w+)*)*\\s*)+', " ", message)


def clearText(message):
    tokens = nltk.word_tokenize(message)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return words

outputPath='src/main/resources/preprocessedData.csv'

def preprocess(url):

    data = pd.read_csv(url)
    data["urlsCount"] = data["Tweet"].apply(urls).apply(len)
    data["Tweet"] = data["Tweet"].apply(removeUrls)
    data["hashtags"] = data["Tweet"].apply(hashtags)
    data["hashtagsCount"] = data["hashtags"].apply(len)
    data["mentions"] = data["Tweet"].apply(mentions)
    data["mentionsCount"] = data["mentions"].apply(len)
    data["Tweet"] = data["Tweet"].apply(removeMentions).apply(removeHashtags)
    data["sentencesCount"] = data["Tweet"].apply(nltk.sent_tokenize).apply(len)
    data["charsCount"] = data["Tweet"].apply(len)
    data["numbersToChars"] = data["Tweet"].apply(lambda x: sum([len(c) for c in x if x.isnumeric()]))/data["charsCount"]
    data["wordsCount"] = data["Tweet"].apply(clearText).apply(len)
    data["hashtagsPercentage"] = data["hashtagsCount"] / (data["wordsCount"] + data["hashtagsCount"])
    data["urlsPercentage"] = data["urlsCount"] / (data["wordsCount"] + data["urlsCount"])
    data["mentionsPercentage"] = data["mentionsCount"] / (data["wordsCount"] + data["mentionsCount"])
    vectorizer = CountVectorizer(tokenizer=clearText, token_pattern=None, max_features=int(1e2))
    x = vectorizer.fit_transform(data["Tweet"])
    x=pd.DataFrame.sparse.from_spmatrix(x)
    data=pd.concat([data,x],axis=1)
    data = data.drop(['Tweet','hashtags','mentions','Id'], axis=1)
    place = LabelEncoder()
    data["location"] = place.fit_transform(data["location"])
    data["Type"] = data["Type"].apply(lambda x: x == "Quality")
    data = data.fillna(0)
    data.to_csv(outputPath, index=False)

preprocess(sys.argv[2])

data = pd.read_csv(outputPath)
x = data.drop('Type', axis=1)
y = data['Type']
h = SelectKBest(chi2, k=5)
new_x = h.fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2)

def test(model, title):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print( f1_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(format(roc_auc_score(y_test, y_pred_prob[:, 1])))
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    DetCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
    plt.title(title)
    plt.show()

if sys.argv[1]=="Naive" :
    model_nb = BernoulliNB()
    test(model_nb, 'Bernoulli Naive bayes')
if sys.argv[1]=="Multi" :
    model_nn = MLPClassifier()
    test(model_nn, 'Multi-layer Perceptron classifier')
if sys.argv[1]=="Tree" :
    qualityTree = tree.DecisionTreeClassifier()
    test(qualityTree, 'Decision Tree classifier')
    i = 0
    plt.barh(h.get_feature_names_out(), qualityTree.feature_importances_)
    plt.title('Best Features ')
    plt.show()
    dot_data = StringIO()
