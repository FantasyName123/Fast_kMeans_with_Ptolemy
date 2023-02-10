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
    print(f'Lloyd {iteration_count}')
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
    indices = np.array(range(n))
    centroids = initial_centroids
    iteration_count = 0
    converged = False
    distances = np.ones((n, k)) * np.inf
    upper_bounds = np.ones(n) * np.inf
    lower_bounds = np.zeros((n, k))
    np.random.seed(0)
    assignment = np.random.default_rng().integers(low=0, high=k, size=n)
    r = np.array([True] * n)

    # numbers as in Elkan's original paper
    while not converged:
    # while iteration_count < 40:
        iteration_count += 1
        if iteration_count > 1:
            converged = True

        # 1
        center_distances = np.array([np.linalg.norm(centroids - centroids[i], axis=1) for i in range(k)])
        center_distances[center_distances == 0] = [np.nan]
        center_bounds = 1/2 * np.nanmin(center_distances, axis=0)

        # 2
        center_condition = center_bounds[assignment] < upper_bounds  # atm list of booleans
        relevant_indices = indices[center_condition]
        relevant_assignment = assignment[center_condition]
        relevant_upper_bound = upper_bounds[center_condition]

        # 3
        for center_index, center in enumerate(centroids):
            # Note:
            # In Elkan's Paper | In this Code
            #        c         |  center_index
            #       c(x)       | assignment[...]
            # (i)
            not_same_center = np.not_equal(relevant_assignment, center_index)  # broadcasting
            # (ii)
            lu_bound_condition = lower_bounds[relevant_indices, center_index] < relevant_upper_bound
            # (iii)
            center_condition_again = 1/2 * center_distances[center_index, relevant_assignment] < relevant_upper_bound

            all_conditions = not_same_center & lu_bound_condition & center_condition_again
            unpruned_indices = relevant_indices[all_conditions]
            # for testing purposes
            # unpruned_indices = indices

            # 3a - d(x,c(x))
            r_condition = r[unpruned_indices]
            need_to_update_indices = unpruned_indices[r_condition]
            nooo_need_to_update_indices = unpruned_indices[np.logical_not(r_condition)]
            # compute distances
            distances[need_to_update_indices, assignment[need_to_update_indices]] = \
                np.linalg.norm(data[need_to_update_indices] - centroids[assignment[need_to_update_indices]], axis=1)
            # update bounds after distance computation
            upper_bounds[need_to_update_indices] = distances[need_to_update_indices, assignment[need_to_update_indices]]
            lower_bounds[need_to_update_indices, assignment[need_to_update_indices]] = \
                distances[need_to_update_indices, assignment[need_to_update_indices]]
            # distance equals upper bound for these indices now
            r[need_to_update_indices] = False

            # if r is False: upper bound is still tight
            distances[nooo_need_to_update_indices, assignment[nooo_need_to_update_indices]] \
                = upper_bounds[nooo_need_to_update_indices]

            # 3b - d(x,c)
            condition_3b = distances[unpruned_indices, assignment[unpruned_indices]] > np.min(
                [lower_bounds[unpruned_indices, center_index],
                 1/2 * center_distances[assignment[unpruned_indices], center_index]], axis=0)
            condition_3b_indices = unpruned_indices[condition_3b]
            # compute distances
            distances[condition_3b_indices, center_index] = np.linalg.norm(data[condition_3b_indices] - center, axis=1)
            new_assignment = distances[unpruned_indices, center_index] \
                             < distances[unpruned_indices, assignment[unpruned_indices]]
            new_assignment_indices = unpruned_indices[new_assignment]
            # update assignment
            assignment[new_assignment_indices] = center_index
            # update upper_bounds when assignment is changed!
            upper_bounds[new_assignment_indices] = distances[new_assignment_indices, center_index]
            if converged and len(new_assignment_indices) > 0:
                converged = False

        # 4
        new_centroids = np.copy(centroids)
        for center_index, center in enumerate(centroids):
            assigned_points = data[assignment == center_index]
            if len(assigned_points) > 0:
                new_centroids[center_index] = np.mean(assigned_points, axis=0)

        centroids_movement = np.linalg.norm(new_centroids - centroids, axis=1)

        # 5
        for center_index, center in enumerate(centroids):
            # lower_bounds[indices, center_index] = np.max((
            #     lower_bounds[indices, center_index] - centroids_movement[center_index], 0), axis=0)
            lower_bounds[indices, center_index] -= centroids_movement[center_index]

        # 6
        upper_bounds[indices] += centroids_movement[assignment]
        r[indices] = True

        # 7
        # check for convergence
        # if np.allclose(centroids, new_centroids):
        #     converged = True
        # else:
        #     centroids = new_centroids
        centroids = new_centroids

    print(f'Elkan {iteration_count}')
    return centroids, assignment








