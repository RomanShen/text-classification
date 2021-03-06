#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: romanshen 
@file: one.py 
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


def new_cts(cluster):
    tmp = []
    for i in range(cluster.shape[0]):
        if np.sum(self.ct_indices == i) == 0:
            tmp.append(self.cts[i])
        else:
            tmp.append(self.X[self.ct_indices == i].mean(axis=0))
    self.cts = np.array(tmp)


class OnePassClustering:
    def __init__(self, X, classes=3):
        self.X = X
        self.classes = classes
        self.distance_matrix = pairwise_distances(self.X, metric='euclidean') * -1
        np.fill_diagonal(self.distance_matrix, np.inf)
        self.array = np.arange(self.distance_matrix.shape[0])
        self.clusters = []
        seaborn.set_style(style='white')
        self.fig, self.ax = plt.subplots()
        self.colors = ['#990066', '#FAEBD7', '#000000', '#0033FF', '#6B8E23', '#EE82EE', '#9ACD32',
                       '#FF00FF', '#F08080', '#9370DB']
        self.threshold = -2.3

    def define(self, index):
        m = -99999
        indd = 0
        for i in range(len(self.clusters)):
            sim = np.sqrt(((self.clusters[i][1] - self.X[index]) ** 2).sum()) * -1
            indd = i
            if sim > m:
                m = sim
        if m > self.threshold:
            self.clusters[indd][0].append(index)
            self.clusters[indd][1] = self.X[self.clusters[indd][0]].mean(axis=0)
        else:
            self.clusters.append([[index], self.X[index]])

    def fit(self, index=0):
        plt.ion()
        # self.clusters.append(list(self.array))

        self.current_cluster = [[index], self.X[index]]
        self.clusters = [self.current_cluster]
        self.no_classify = [i for i in range(10)]
        self.no_classify.remove(index)
        while self.no_classify:
            print("-----------")
            index = self.no_classify.pop(0)
            self.define(index)
            # print(self.clusters)
            # print("\n")

        # plt.savefig(f"./one/hie_mean_{i}.png")
        # plt.pause(1.0)
        # plt.ioff()
        # plt.show()
        for i in range(len(self.clusters)):
            print(self.clusters[i][0])

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
    hie = OnePassClustering(data)
    hie.fit()

    # eng_data = generate_eng_bow('./english_data.txt')
    # eng_hie = HierarchicalMeanClustering(eng_data)
    # eng_hie.fit()
