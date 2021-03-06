#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: romanshen 
@file: scatter.py.py 
@time: 2021/03/05
@contact: xiangqing.shen@njust.edu.cn
"""
import matplotlib.pyplot as plt


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
