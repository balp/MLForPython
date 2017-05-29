import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
from sklearn.datasets.samples_generator import make_blobs

centers=random.randrange(2,8)
print(centers)
X,y = make_blobs(n_samples=50, centers=centers, n_features=2)

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3],])

#plt.scatter(X[:,0], X[:,1], s=150)
#6plt.show()

colors = 10*["g","r","c","b","k"]


class MeanShift:
    def __init__(self, radius=None, radius_norm_steps=100):
        self._radius = radius
        self._radius_norm_steps = radius_norm_steps

    def fit(self, data):
        centroids = {}
        if self._radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self._radius = all_data_norm / self._radius_norm_steps

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self._radius_norm_steps)][::-1]
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for feature_set in data:
                    distance = np.linalg.norm(feature_set-centroid)
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self._radius)
                    if weight_index > self._radius_norm_steps-1:
                        weight_index = self._radius_norm_steps-1
                    to_add = (weights[weight_index]**2)*[feature_set]
                    in_bandwidth += to_add
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centroids)))
            to_pop = set()
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self._radius:
                        to_pop.add(ii)
                        break
            for i in to_pop:
                uniques.remove(i)

            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break
            if optimized:
                break
        self.centroids = centroids
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for feature_set in data:
            distances = [np.linalg.norm(feature_set-self.centroids[centroid])
                         for centroid in centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(feature_set)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in centroids]
        classification = distances.index(min(distances))
        return classification

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='x', color=color,
                    s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()


