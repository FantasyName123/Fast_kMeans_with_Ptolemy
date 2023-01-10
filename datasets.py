import numpy as np
from sklearn.datasets import make_blobs


def create_dataset_uniform(size=10000, dimension=2) -> list[tuple]:
    """
    :param size: the number of data objects to be returned
    :param dimension: the number of coordinates of each data object
    :return: list of tuples, each tuple representing one data object
    """
    uniform_data = np.random.rand(size, dimension)
    uniform_data *= 100
    uniform_data = [tuple(row) for row in uniform_data]

    return uniform_data


# todo: weitere Datasets erstellen
# kann auch Punkte außerhalb von [0, 100]^n enthalten!
# Ist aber wahrscheinlich erstmal nicht weiter schlimm
def create_clustered_data(size=10000, dimension=2, clusters=5) -> list[tuple]:
    """
    :param size: the number of data objects to be returned
    :param dimension: the number of coordinates of each data object
    :param clusters: the number of clusters
    :return: list of tuples, each tuple representing one data object
    """
    # n_features = dimension
    data, assignment = make_blobs(n_samples=size, n_features=dimension, centers=clusters,
                                  center_box=(0, 100), cluster_std=np.sqrt(10) + 0.5, random_state=1)
    clustered_data = [tuple(row) for row in data]

    return clustered_data


