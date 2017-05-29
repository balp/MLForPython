import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

colors = 10 * ["g.", "r.", "c.", "b.", "k."]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self._k = k
        self._tol = tol
        self._max_iter = max_iter

    def fit(self, data):
        self._centroids = []

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
                pass
                # self._centroids[classification] \
                #    = np.average(self._classifications[classification], axis=0)

    def predict(self, data):
        pass


# clf = KMeans(n_clusters=4)
# clf.fit(X)
# centroids = clf.cluster_centers_
# labels = clf.labels_


# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
# plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidth=5)
#
# plt.show()

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5, zorder=10)
plt.show()
