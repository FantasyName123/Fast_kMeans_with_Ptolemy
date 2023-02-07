import numpy as np

from subroutines import *
from helpers import *


# todo: There are two approaches I see right now: 1) Putting the label as the last entry in each point's array
#  2) creating a separate array with all the labels -> have to be careful that the order is always the same,
#  but should be no problem after all
def lloyd_matrix(data, initial_centroids):
    """
    With help of ChatGPT...

    :param data:
    :param initial_centroids:
    :return: centroids, clusters
    """
    # initialise
    k = len(initial_centroids)
    n = len(data)
    centroids = initial_centroids
    iteration_count = 0
    converged = False

    while not converged:
        iteration_count += 1
        # assign each point to the nearest centroid
        distances = np.array([np.linalg.norm(data - centroids[i], axis=1) for i in range(k)])
        clusters = np.argmin(distances, axis=0)

        # update centroids
        new_centroids = np.copy(centroids)
        for i in range(k):
            if np.sum(clusters == i) > 0:
                new_centroids[i] = np.mean(data[clusters == i], axis=0)

        # check for convergence
        if np.allclose(centroids, new_centroids):
            converged = True
        else:
            centroids = new_centroids

    return centroids, clusters


def elkan_matrix(data, initial_centroids):
    """

    :param data:
    :param initial_centroids:
    :return: centroids, clusters
    """
    # initialise
    k = len(initial_centroids)
    n = len(data)
    centroids = initial_centroids
    iteration_count = 0
    converged = False
    upper_bounds = np.ones(n) * np.inf
    lower_bounds = np.zeros((n, k))
    # todo: initialise assignment
    assignment = np.random.default_rng().integers(low=0, high=k, size=n)

    # numbers as in Elkan's original paper
    while not converged:
        iteration_count += 1

        # 1
        center_distances = np.array([np.linalg.norm(centroids - centroids[i], axis=1) for i in range(k)])
        center_distances[center_distances == 0] = [np.nan]
        center_bounds = 1/2 * np.nanmin(center_distances, axis=0)

        # 2
        relevant_indices = center_bounds[assignment] < upper_bounds  # atm list of booleans

        # 3
        for idx, center in enumerate(centroids):
            # (i)
            not_same_center = np.not_equal(assignment, idx)  # broadcasting
            # (ii)
            lu_bound_condition = lower_bounds[:, idx] < upper_bounds
            # (iii)
            center_bound_condition = 1/2 * center_distances[idx, assignment] < upper_bounds


        # assign each point to the nearest centroid
        distances = np.array([np.linalg.norm(data - centroids[i], axis=1) for i in range(k)])
        clusters = np.argmin(distances, axis=0)

        # update centroids
        new_centroids = np.copy(centroids)
        for i in range(k):
            if np.sum(clusters == i) > 0:
                new_centroids[i] = np.mean(data[clusters == i], axis=0)

        # check for convergence
        if np.allclose(centroids, new_centroids):
            converged = True
        else:
            centroids = new_centroids

    return centroids, clusters


def k_means_elkan(X, initial_centroids, max_iter=100):
    N, d = X.shape
    k = len(initial_centroids)

    # randomly initialize centroids
    centroids = initial_centroids

    # initialize lower bounds between points and centroids
    lower_bounds = np.zeros((N, k))

    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        # update lower bounds
        for i in range(k):
            lower_bounds[:, i] = np.linalg.norm(X - centroids[i], axis=1) ** 2

        # assign each point to the nearest centroid
        clusters = np.argmin(lower_bounds, axis=1)

        # update centroids
        new_centroids = np.copy(centroids)
        for i in range(k):
            if np.sum(clusters == i) > 0:
                X_cluster = X[clusters == i]
                distances = np.linalg.norm(X_cluster - centroids[i], axis=1) ** 2
                lower_bound = np.min(distances)
                upper_bound = np.max(distances)

                if lower_bound >= lower_bounds[clusters == i, i]:
                    new_centroids[i] = np.mean(X_cluster, axis=0)
                else:
                    lower_bounds[clusters == i, i] = upper_bound

        # check for convergence
        if np.allclose(centroids, new_centroids):
            converged = True
        else:
            centroids = new_centroids
            iteration += 1

    return centroids, clusters






