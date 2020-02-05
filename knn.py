import os
from nltk.corpus import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
from parameters import *
from sklearn.metrics.pairwise import cosine_similarity


class knn():
    def __init__(self, train_x, train_y, test_x, test_y, distance="COS"):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.distance = distance

    def euclidean(self, test):
        train_x = np.array(self.train_x)
        dis_diff_list = np.sum((test - train_x) ** 2, axis=1)
        sorted_index = dis_diff_list.argsort()[:k]
        return sorted_index

    def cosine_sim(self, test):
        train_x = np.array(self.train_x)
        test = np.array([test])
        cos_sim = cosine_similarity(train_x, test)
        sorted = cos_sim.argsort(axis=0)[-k:]

        flat_list = [item for sublist in sorted for item in sublist]
        return flat_list

    def get_top_k(self, y):
        # Gets the top k nearest neighbors
        if self.distance == "COS":
            sorted_index = self.cosine_sim(y)
        else:
            sorted_index = self.euclidean(y)
        return sorted_index

    def prediction(self, k_neighbors):
        # Gets the final prediction based on the top k neighbors
        list_predictions = np.array(self.train_y)[k_neighbors]
        s = sum(list_predictions)
        if s > k / 2:
            pred = 1
        else:
            pred = 0
        return pred

    def accuracy(self, predicted_y):
        num_accurate = sum(np.array(self.test_y) == np.array(predicted_y))
        percentage_accuracy = (num_accurate / len(self.test_y)) * 100
        return percentage_accuracy


def to_vector(x, vectorizer):
    # Converts words into vectors based on the vectorizer provided
    final_x = []

    for i in x:
        vector = vectorizer.transform([i])
        vector = vector.toarray()[0]
        vector = list(vector)
        final_x.append(vector)
    return final_x


def main(train_x, test_x, train_y, test_y):
    global k
    # create a TfidfVectorizer and fit it
    vectorizer = TfidfVectorizer(max_features=100, norm=None)
    vectorizer.fit(train_x)

    # Convert train and test sets to vectors
    final_train_x = to_vector(train_x, vectorizer)
    final_test_x = to_vector(test_x, vectorizer)
    predicted_y = []

    # call the class knn and use it to predict the test sets
    class_knn = knn(final_train_x, train_y, final_test_x, test_y, distance=DISTANCE)
    for test in final_test_x:
        k_neighbors = class_knn.get_top_k(test)
        predicted_y.append(class_knn.prediction(k_neighbors))

    print("Accuracy:", class_knn.accuracy(predicted_y))




