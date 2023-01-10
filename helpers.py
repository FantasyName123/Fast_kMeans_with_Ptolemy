import numpy as np
import pandas as pd
from math import dist


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