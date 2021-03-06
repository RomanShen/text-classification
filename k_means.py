#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: romanshen
@file: k_means.py
@time: 2021/03/05
@contact: xiangqing.shen@njust.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocess_eng import generate_bow
from scatter import mscatter


class KMeans:
    def __init__(self, X, ct_indices=[1, 4, 6]):
        self.X = X
        self.cts = X[ct_indices]
        self.radius = np.zeros(self.cts.shape[0])
        self.ct_indices = np.zeros(self.X.shape[0])
        self.clusters = []
        seaborn.set_style(style='white')
        self.fig, self.ax = plt.subplots()
        self.markers = {0: 'o', 1: 's', 2: '^'}
        self.interclass_sum = []

    def fit(self, iter_times=4):
        pca = PCA(n_components=2)
        standard_x = StandardScaler().fit_transform(self.X)
        pca_X = pca.fit_transform(standard_x)

        for i in range(iter_times):
            self.ax.cla()
            self.ax.xaxis.set_major_locator(MultipleLocator(2))
            self.ax.yaxis.set_major_locator(MultipleLocator(2))
            plt.xlim(-6, 6)
            plt.ylim(-6, 6)

            self.calculate_radius()
            self.clusters.append(self.ct_indices)

            cm = list(map(lambda x: self.markers[x], self.ct_indices.tolist()))

            # self.ax.set_title("KMeans: Step %d, Initial Status" % i)

            scatter = mscatter(pca_X[:, 0], pca_X[:, 1], m=cm, c='black', ax=self.ax)

            for l in range(self.X.shape[0]):
                self.ax.text(pca_X[l, 0] + 0.08, pca_X[l, 1] - 0.08, l + 1, verticalalignment='center',
                                horizontalalignment='left')
            plt.savefig(f"./kmeans/kmeans_{i}.png")

            plt.pause(0.5)

            self.new_cts()

        plt.ioff()
        plt.show()

        print(self.clusters)

    def calculate_radius(self):
        dist = np.sqrt(((self.X - self.cts[:, np.newaxis]) ** 2).sum(axis=2))
        self.ct_indices = np.argmin(dist, axis=0)
        dist = dist.min(axis=0)
        self.interclass_sum = np.sum(dist)

    def new_cts(self):
        tmp = []
        for i in range(self.cts.shape[0]):
            if np.sum(self.ct_indices == i) == 0:
                tmp.append(self.cts[i])
            else:
                tmp.append(self.X[self.ct_indices == i].mean(axis=0))
        self.cts = np.array(tmp)


if __name__ == '__main__':
    data = generate_bow('./vec_eng_data.txt')

    km = KMeans(data, [1, 4, 6])
    # # km = KMeans(data, [1, 4, 7])
    # # km = KMeans(data, [2, 4, 7])
    km.fit()

    # from sklearn.cluster import KMeans
    # y_pred = KMeans(n_clusters=3, init=data[[1, 4, 7]]).fit_predict(data)
    # print(y_pred)