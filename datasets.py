import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def create_dataset_uniform(size=10000, dimension=2) -> list[tuple]:
    """
    Creates a data set of the given size, where the data points are uniformly random distributed in the
    cube [0,100]^dimension.

    :param size: The number of data objects to be returned.
    :param dimension: The number of coordinates of each data object.
    :return: List of tuples, each tuple representing one data object.
    """
    uniform_data = np.random.rand(size, dimension)
    uniform_data *= 100
    uniform_data = [tuple(row) for row in uniform_data]

    return uniform_data


def create_clustered_data(size=10000, dimension=2, clusters=5) -> list[tuple]:
    """
    Creates a clustered data set of the given size. Uses sklearn.datasets.make_blobs to do so. The cluster standard
    deviation is set to 10 and a random state is fixed, for reproducibility. The bounding box for the centers is set to
    the cube [0,100]^dimension: This means the data points can lie outside that box.

    :param size: The number of data objects to be returned.
    :param dimension: The number of coordinates of each data object.
    :param clusters: The number of clusters.
    :return: List of tuples, each tuple representing one data object.
    """
    data, assignment = make_blobs(n_samples=size, n_features=dimension, centers=clusters,
                                  center_box=(0, 100), cluster_std=np.sqrt(10), random_state=1)
    clustered_data = [tuple(row) for row in data]

    return clustered_data


def birch_1_to_pickle():
    """
    This method was used to transform the Birch1 from the rtf format into the pickle format. It is not needed anymore,
    but still here, in case it is needed again or as a template for other datasets.

    :return: None.
    """
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
    """
    Retrieves the Birch_1 dataset. It consists of 100,000 data points associated to 100 clusters. More information about
    the Birch_1 dataset at: http://cs.joensuu.fi/sipu/datasets/

    :return: The Birch_1 dataset as a list of tuples.
    """
    df = pd.read_pickle('Birch_1_dataset')
    data = [tuple(row) for index, row in df.iterrows()]
    return data


def get_birch_2_subset(size):
    """
    Retrieves a Birch_2 subset of adjustable size. The number of clusters is fixed as 100. More information about the
    Birch_2 dataset at: http://cs.joensuu.fi/sipu/datasets/

    :param size: Integer between 1 and 99 (boundaries included). The actual number of data points will be 1000 times the
        passed size.
    :return: A Birch_2 dataset as a list of tuples.
    """
    url_string = f'b2-random-txt/b2-random-{size}.txt'
    with open(url_string) as file:
        lines = file.readlines()
        lines = [line.replace('\n', '').split() for line in lines]
        lines = [tuple([int(item) for item in new_line]) for new_line in lines]
        print(f'Length of data set: {len(lines)}')

    return lines
