import numpy as np
import pandas as pd
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
                                  center_box=(0, 100), cluster_std=np.sqrt(10), random_state=1)
    clustered_data = [tuple(row) for row in data]

    return clustered_data


def birch_1_to_pickle():
    with open('Birch_1.rtf') as f:
        lines = f.readlines()
    lines_numbers = lines[9:-1]
    lines_numbers[0] = lines_numbers[0].replace('\\outl0\\strokewidth0 \\strokec2', '')
    lines_list = []
    for idx in range(len(lines_numbers)):
        new_line = lines_numbers[idx]
        new_line = new_line.replace('\\\n', '')
        new_line = new_line.split()
        new_line = [int(item) for item in new_line]
        lines_list.append(new_line)

    df = pd.DataFrame(lines_list)
    df.to_pickle('/Users/louisborchert/PycharmProjects/Fast_kMeans_with_Ptolemy/Birch_1_dataset.pkl')


def get_birch_1():
    df = pd.read_pickle('Birch_1_dataset')
    a = df.index.values
    data = [tuple(row) for index, row in df.iterrows()]
    return data


def get_birch_2_1(url_string='/Users/louisborchert/Downloads/b2-random-txt/b2-random-10.txt'):
    with open(url_string) as file:
        lines = file.readlines()
        lines = [line.replace('\n', '').split() for line in lines]
        lines = [tuple([int(item) for item in new_line]) for new_line in lines]
        print(f'Length of data set: {len(lines)}')

    return lines


def get_test_dataset():
    test_data = [(1, 1), (1, 3), (1, 5), (2, 1), (2, 4),
                 (4, 4), (4, 5), (4, 6), (5, 5), (5, 6),
                 (10, 9), (11, 12), (11, 14), (12, 15)]
    test_data = np.array(test_data)
    return test_data

