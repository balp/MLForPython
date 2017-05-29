#!/usr(bin/env python3.5
#
#
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import profile
from multiprocessing import Pool, TimeoutError

style.use('ggplot')

class SupportVectorMachine:
    def __init__(self, visualization=True):
        self._visualization = visualization
        self._colors = {1:'r', -1:'b'}
        if self._visualization:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(1,1,1)
        self._w = np.nan
        self._b = np.nan

        self._max_feature_value = -np.infty
        self._min_feature_value = np.infty

    def fit(self, data):
        self._data = data
        self._feature_min_max()

        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
        step_sizes = [self._max_feature_value * 0.1,
                      self._max_feature_value * 0.01,
                      self._max_feature_value * 0.001]
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self._max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            self._opt_choice = {}
            while not optimized:
                for b in np.arange(-1*(self._max_feature_value*b_range_multiple),
                                self._max_feature_value*b_range_multiple,
                                step*b_multiple):
                    for transform in transforms:
                        self._test_transform_on_all_data(b, transform, w)
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w - step
            norms = sorted([n for n in self._opt_choice])
            opt_choice = self._opt_choice[norms[0]]
            self._w = opt_choice[0]
            self._b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def _test_transform_on_all_data(self, b, transform, w):
        w_t = w * transform
        found_option = True
        for i in self._data:
            for xi in self._data[i]:
                yi = i
                if not yi * (np.dot(w_t, xi) + b) >= 1:
                    found_option = False
                    break
            if not found_option:
                break
        if found_option:
            self._opt_choice[np.linalg.norm(w_t)] = [w_t, b]

    def _feature_min_max(self):
        for yi in self._data:
            for feature_set in self._data[yi]:
                for feature in feature_set:
                    self._max_feature_value = max(self._max_feature_value, feature)
                    self._min_feature_value = min(self._min_feature_value, feature)


    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self._w) + self._b)
        if classification != 0 and self._visualization:
            self._ax.scatter(features[0], features[1], s=200, marker='*', c=self._colors[classification])
        return classification

    def visualize(self):
        def _hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        [[self._ax.scatter(x[0], x[1], s=100, color=self._colors[i]) for x in self._data[i]] for i in self._data]
        data_range = (self._min_feature_value*0.9, self._max_feature_value*1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        psv1 = _hyperplane(hyp_x_min, self._w, self._b, 1)
        psv2 = _hyperplane(hyp_x_max, self._w, self._b, 1)
        self._ax.plot([hyp_x_min, hyp_x_max ], [psv1, psv2], 'k')

        nsv1 = _hyperplane(hyp_x_min, self._w, self._b, -1)
        nsv2 = _hyperplane(hyp_x_max, self._w, self._b, -1)
        self._ax.plot([hyp_x_min, hyp_x_max ], [nsv1, nsv2], 'k')

        db1 = _hyperplane(hyp_x_min, self._w, self._b, 0)
        db2 = _hyperplane(hyp_x_max, self._w, self._b, 0)
        self._ax.plot([hyp_x_min, hyp_x_max ], [db1, db2], 'y--')

        plt.show()

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

if __name__ == '__main__':
    svm = SupportVectorMachine()
    profile.run('svm.fit(data_dict)')
    predict_us = [[0, 10],
                  [1, 3],
                  [3, 4],
                  [3, 5],
                  [5, 5],
                  [5, 6],
                  [6, -5],
                  [5, 8]]
    for p in predict_us:
        svm.predict(p)
    svm.visualize()