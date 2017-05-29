import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

colors = 10 * ["g", "r", "c", "b", "k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self._k = k
        self._tol = tol
        self._max_iter = max_iter

    def fit(self, data):
        self._centroids = {}

        for i in range(self._k):
            self._centroids[i] = data[i]

        for i in range(self._max_iter):
            self._classifications = {}
            for i in range(self._k):
                self._classifications[i] = []
            for featureset in data:
                distances = [np.linalg.norm(featureset
                                            - self._centroids[centroid])
                             for centroid in self._centroids]
                classification = distances.index(min(distances))
                self._classifications[classification].append(featureset)

            prev_centroids = dict(self._centroids)
            for classification in self._classifications:
                self._centroids[classification] \
                    = np.average(self._classifications[classification], axis=0)
            optimized = True
            for c in self._centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self._centroids[c]
                if np.sum((current_centroid - original_centroid)
                                  / original_centroid * 100.0) > self._tol:
                    optimized = False
                    break
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data
                                    - self._centroids[centroid])
                     for centroid in self._centroids]
        classification = distances.index(min(distances))
        return classification

df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_values = {}
        def convert_to_int(val):
            return text_digit_values[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df
df = handle_non_numerical_data(df)

df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])


clf = K_Means()
clf.fit(X)
centroids = clf._centroids


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

correct_rate = correct / len(X)
if correct_rate < 0.5:
    correct_rate = 1 - correct_rate
print(correct_rate * 100)
