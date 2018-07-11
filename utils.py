# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# sys.path.append('/Users/chuxing/python/test');
from Datasets import Dataset
from math import radians, cos, sin, asin, sqrt


def haversine(v1, v2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    lon1, lat1, lon2, lat2 = map(radians, [v1[1], v1[0], v2[1], v2[0]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return float(c * r)


def ec_dis(v1, v2):
    """
    calculate the Euclidean distance
    """
    return np.sqrt(np.sum(np.square(v1 - v2)))


def distribution_user_view():
    """
    view the distribution of distance
    """

    dataset = Dataset()
    data = dataset.data
    label = dataset.label
    data = data.reshape((len(data), -1))

    data_nagitive = []
    data_positive = []
    for i in range(len(label)):
        item = data[i]
        if label[i] == 0.0:
            for j in range(len(item)):
                if item[j] != -1.0:
                    data_nagitive.append(float(item[j]))
        else:
            for j in range(len(item)):
                if item[j] != -1.0:
                    data_positive.append(float(item[j]))
    print(type(data_positive[0]))

    sns.kdeplot(data_positive, color='r', )
    sns.kdeplot(data_nagitive, color='b', )
    plt.show()


if __name__ == '__main__':
    distribution_user_view()
