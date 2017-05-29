import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

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

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
# plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5, zorder=10)
# plt.show()

clf = K_Means()
clf.fit(X)
centroids = clf._centroids
# labels = clf.labels_


for centroid in clf._centroids:
    plt.scatter(clf._centroids[centroid][0], clf._centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf._classifications:
    color = colors[classification]
    print(color)
    for featureset in clf._classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color,
                    s=150, linewidths=5)

unknowns = np.array([[1, 3],
                     [2, 8],
                     [2, 9],
                     [0, 3],
                     [6, 4],
                     [5, 3],
                     ])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*',
                color=colors[classification], s=150, linewidths=5)

plt.show()
