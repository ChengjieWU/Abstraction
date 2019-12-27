import random

import numpy as np
from pyemd import emd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric


distance_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        distance_matrix[i][j] = abs(i - j)


def emd_distance(x, y):
    return emd(x, y, distance_matrix)


if __name__ == "__main__":
    a = np.load("/home/cj/data/distribution_turn.npy")
    indices = random.sample(list(range(a.shape[0])), 5000000)
    a_s = a[indices, :]

    metric = distance_metric(type_metric.USER_DEFINED, func=emd_distance)
    initial_centers = kmeans_plusplus_initializer(a_s, 200).initialize()
    kmeans_instance = kmeans(a_s, initial_centers, metric=metric)

    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()
