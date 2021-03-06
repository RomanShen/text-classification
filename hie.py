#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: romanshen 
@file: hie.py 
@time: 2021/03/05
@contact: xiangqing.shen@njust.edu.cn
"""


import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from preprocess_eng import generate_bow

import seaborn


class HierarchicalMeanClustering:
    def __init__(self, X, classes=3):
        self.X = X
        self.classes = classes
        self.distance_matrix = pairwise_distances(self.X, metric='cosine')
        np.fill_diagonal(self.distance_matrix, np.inf)
        self.array = np.arange(self.distance_matrix.shape[0])
        self.clusters = []
        seaborn.set_style(style='white')
        self.fig, self.ax = plt.subplots()
        self.colors = ['#990066', '#FAEBD7', '#000000', '#0033FF', '#6B8E23', '#EE82EE', '#9ACD32',
                       '#FF00FF', '#F08080', '#9370DB']

    def fit(self):
        plt.ion()
        self.clusters.append(list(self.array))
        i = 1
        while True:
            self.ax.cla()
            self.ax.xaxis.set_major_locator(MultipleLocator(2))
            self.ax.yaxis.set_major_locator(MultipleLocator(2))
            plt.xlim(-8, 8)
            plt.ylim(-8, 8)
            self.ax.set_title(f"Hierarchical Clustering : Step {i}")

            pca = PCA(n_components=2)
            standard_x = StandardScaler().fit_transform(self.X)
            pca_X = pca.fit_transform(standard_x)

            # pca_X = pca(self.X)[0]

            self.ax.scatter(pca_X[:, 0], pca_X[:, 1], c=[
                            self.colors[i] for i in self.array], s=40)
            for l in range(self.X.shape[0]):
                self.ax.text(pca_X[l, 0] + 0.08, pca_X[l, 1], l + 1,
                             verticalalignment='center', horizontalalignment='left')

            plt.savefig(f"./hie/hie_mean_{i}.png")
            i += 1
            plt.pause(1.5)
            self.find_cluster()
            if np.unique(self.clusters[-1]).shape[0] < self.classes:
                break
        # self.ax.set_title(f"Hierarchical Clustering : Step {i}")
        # self.ax.scatter(pca(self.X)[0][:, 0], pca(self.X)[0][:, 1], c=[
        #                 self.colors[i] for i in self.array], s=40)
        # for l in range(self.X.shape[0]):
        #     self.ax.text(pca_X[l, 0] + 0.08, pca_X[l, 1], l + 1,
        #                  verticalalignment='center', horizontalalignment='left')
        plt.savefig(f"./hie/hie_mean_{i}.png")
        plt.pause(1.0)
        plt.ioff()
        plt.show()
        print(self.clusters)

    def find_cluster(self):
        row_index = -1
        col_index = -1
        min_val = np.inf
        for i in range(0, self.distance_matrix.shape[0]):
            for j in range(0, self.distance_matrix.shape[1]):
                if self.distance_matrix[i][j] <= min_val:
                    min_val = self.distance_matrix[i][j]
                    row_index = i
                    col_index = j
        for i in range(0, self.distance_matrix.shape[0]):
            if i != col_index:
                temp = (self.distance_matrix[col_index][i] +
                        self.distance_matrix[row_index][i]) / 2
                self.distance_matrix[col_index][i] = temp
                self.distance_matrix[i][col_index] = temp

        for i in range(0, self.distance_matrix.shape[0]):
            self.distance_matrix[row_index][i] = np.inf
            self.distance_matrix[i][row_index] = np.inf

        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(self.array)):
            if self.array[n] == maximum:
                self.array[n] = minimum
        self.clusters.append(list(self.array))


if __name__ == '__main__':
    data = generate_bow('./vec_eng_data.txt')
    hie = HierarchicalMeanClustering(data)
    hie.fit()

    # eng_data = generate_eng_bow('./english_data.txt')
    # eng_hie = HierarchicalMeanClustering(eng_data)
    # eng_hie.fit()
