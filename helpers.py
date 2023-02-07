import numpy as np
import pandas as pd
from math import dist
from sklearn.cluster import KMeans
import time


def all_dist(dataset, query, return_as='Series'):
    """
    :param dataset:
    :param query:
    :param return_as:
    :return:
    """
    distances = np.sqrt(((np.array(dataset) - np.array(query)) ** 2).sum(axis=1))
    if return_as == 'Series':
        # return pd.Series(distances, index=pd.MultiIndex.from_tuples(dataset))
        # return pd.Series(distances, index=dataset)
        return pd.Series(distances)
    elif return_as == 'DataFrame':
        return pd.DataFrame({'objects': list(dataset), 'distances': list(distances)})


def test_sklearn(data, k):
    start = time.time()
    km = KMeans(n_clusters=k, algorithm='elkan').fit(data)
    end = time.time()
    print(f'sklearn needed {round(end - start, 6)} seconds')
    assignment = dict()
    lables = km.labels_
    for idx in range(len(data)):
        data_point = data[idx]
        assignment[data_point] = lables[idx]

    return assignment, km.cluster_centers_


def into_2d_array(data):
    """
    Takes a list of tuples and transforms it into a 2-dimensional numpy array
    :param data: list of tuples
    :return: 2-dimensional numpy array
    """
    return np.array(data)
