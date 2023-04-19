import numpy as np
import pandas as pd
from math import dist
from sklearn.cluster import KMeans
import time


# todo: Methoden mit/ohne Docstring
#  all_dist: Mit
#  test_sklearn: Mit

def all_dist(dataset, query, return_as='Series'):
    """
    Computes the euclidean distance from the query to each point of the dataset. Transforms both into a numpy.ndarray
    in the process.

    :param dataset: 2-dimensional array-like.
    :param query: 1-dimensional array_like.
    :param return_as: 'Series' (default) or 'DataFrame'.
    :return:
    """
    distances = np.sqrt(((np.array(dataset) - np.array(query)) ** 2).sum(axis=1))
    if return_as == 'Series':
        return pd.Series(distances)
    elif return_as == 'DataFrame':
        return pd.DataFrame({'objects': list(dataset), 'distances': list(distances)})
    else:
        raise KeyError(f'{return_as} is not a valid return_as argument')


def test_sklearn(data, k):
    """
    This method is used as a benchmark test for the kMeans implementations in this project. Especially, it is meant to
    check, whether the self-made implementations of existing algorithms has at least a comparable runtime to existing
    implementations of those algorithms.

    :param data: List of tuples. The dataset to test sklearn's implementation of kMeans on.
    :param k: The number of clusters.
    :return:
    """
    km = KMeans(n_clusters=k, algorithm='elkan')
    start = time.time()
    result = km.fit(data)
    end = time.time()
    print(f'sklearn needed {round(end - start, 6)} seconds')

    return result.labels_, result.cluster_centers_
