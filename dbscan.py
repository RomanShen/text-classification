#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: romanshen 
@file: dbscan.py 
@time: 2021/03/05
@contact: xiangqing.shen@njust.edu.cn
"""

from sklearn.cluster import DBSCAN

from preprocess_eng import generate_bow

if __name__ == '__main__':
    data = generate_bow('./vec_eng_data.txt')

    db = DBSCAN(eps=0.6, min_samples=2, metric="cosine").fit(data)
    print(db.labels_)

    # from sklearn.cluster import KMeans
    # y_pred = KMeans(n_clusters=3, init=data[[1, 4, 7]]).fit_predict(data)
    # print(y_pred)