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


def elkan_matrix_full_pto(data, initial_centroids):
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
    loose_upper_bound = np.array([True] * n)  # corresponds to "r" in Elkan's paper
    # ptolemy
    dim = len(data[0])
    pivot = np.zeros(dim)
    pivot_dist_data = np.linalg.norm(data - pivot, axis=1)

    # numbers as in Elkan's original paper
    while not converged:
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

            # 3a - d(x,c(x))
            r_condition = loose_upper_bound[unpruned_indices]
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
            loose_upper_bound[need_to_update_indices] = False

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

        # ptolemy
        pivot_dist_old_center = np.linalg.norm(centroids - pivot, axis=1)
        pivot_dist_new_center = np.linalg.norm(new_centroids - pivot, axis=1)

        # 5
        for center_index, center in enumerate(centroids):
            # lower_bounds[indices, center_index] = np.max((
            #     lower_bounds[indices, center_index] - centroids_movement[center_index], 0), axis=0)
            # lower_bounds[indices, center_index] -= centroids_movement[center_index]
            lower_bounds[indices, center_index] = \
                (lower_bounds[indices, center_index] * pivot_dist_new_center[center_index] -
                 pivot_dist_data[indices] * centroids_movement[center_index]) / pivot_dist_old_center[center_index]
            # erste Laufzeituntersuchungen mit diesen unteren Schranken sehen schlecht aus

        # 6
        # upper_bounds[indices] += centroids_movement[assignment]
        upper_bounds[indices] = ((upper_bounds[indices] * pivot_dist_new_center[assignment] +
                                 pivot_dist_data[indices] * centroids_movement[assignment]) /
                                 pivot_dist_old_center[assignment])
        loose_upper_bound[indices] = True

        # 7
        # check for convergence
        # if np.allclose(centroids, new_centroids):
        #     converged = True
        # else:
        #     centroids = new_centroids
        centroids = new_centroids

    print(f'Elkan Pto {iteration_count}')
    return centroids, assignment


# todo: eine extra Methode, die alle möglichen Zählungen vornimmt für Dreiecks- wie Ptolomäische Unlgeichung
# todo: vielleicht eher die konkrete Anzahl an Distanzberechnungen zählen statt eingesparte (und zusätzliche)
def elkan_matrix_counter(data, initial_centroids):
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
    loose_upper_bound = np.array([True] * n)  # corresponds to "r" in Elkan's paper
    # dist_comp counter (theoretically, not practically, i.e. the implementation needs more than counted)
    dist_comp_tri = 0

    # numbers as in Elkan's original paper
    while not converged:
        iteration_count += 1
        if iteration_count > 1:
            converged = True

        # 1
        center_distances = np.array([np.linalg.norm(centroids - centroids[i], axis=1) for i in range(k)])
        center_distances[center_distances == 0] = [np.nan]
        center_bounds = 1 / 2 * np.nanmin(center_distances, axis=0)
        dist_comp_tri += k * (k-1) / 2

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
            center_condition_again = 1 / 2 * center_distances[center_index, relevant_assignment] < relevant_upper_bound

            all_conditions = not_same_center & lu_bound_condition & center_condition_again
            unpruned_indices = relevant_indices[all_conditions]

            # 3a - d(x,c(x))
            r_condition = loose_upper_bound[unpruned_indices]
            need_to_update_indices = unpruned_indices[r_condition]
            nooo_need_to_update_indices = unpruned_indices[np.logical_not(r_condition)]
            # compute distances
            distances[need_to_update_indices, assignment[need_to_update_indices]] = \
                np.linalg.norm(data[need_to_update_indices] - centroids[assignment[need_to_update_indices]], axis=1)
            dist_comp_tri += len(need_to_update_indices)
            # update bounds after distance computation
            upper_bounds[need_to_update_indices] = distances[need_to_update_indices, assignment[need_to_update_indices]]
            lower_bounds[need_to_update_indices, assignment[need_to_update_indices]] = \
                distances[need_to_update_indices, assignment[need_to_update_indices]]
            # distance equals upper bound for these indices now
            loose_upper_bound[need_to_update_indices] = False

            # if loose_upper_bound is False: upper bound is still tight
            distances[nooo_need_to_update_indices, assignment[nooo_need_to_update_indices]] \
                = upper_bounds[nooo_need_to_update_indices]

            # 3b - d(x,c)
            condition_3b = distances[unpruned_indices, assignment[unpruned_indices]] > np.min(
                [lower_bounds[unpruned_indices, center_index],
                 1 / 2 * center_distances[assignment[unpruned_indices], center_index]], axis=0)
            condition_3b_indices = unpruned_indices[condition_3b]
            # compute distances
            distances[condition_3b_indices, center_index] = np.linalg.norm(data[condition_3b_indices] - center, axis=1)
            dist_comp_tri += len(condition_3b_indices)
            # check, which points need a new assignment
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
        dist_comp_tri += k

        # 5
        for center_index, center in enumerate(centroids):
            # lower_bounds[indices, center_index] = np.max((
            #     lower_bounds[indices, center_index] - centroids_movement[center_index], 0), axis=0)
            lower_bounds[indices, center_index] -= centroids_movement[center_index]

        # 6
        upper_bounds[indices] += centroids_movement[assignment]
        loose_upper_bound[indices] = True

        # 7
        # check for convergence
        # if np.allclose(centroids, new_centroids):
        #     converged = True
        # else:
        #     centroids = new_centroids
        centroids = new_centroids

    print(f'Elkan Dist_Comp {dist_comp_tri}')
    return centroids, assignment


def elkan_matrix_full_ptolemy_counter(data, initial_centroids):
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
    loose_upper_bound = np.array([True] * n)  # corresponds to "r" in Elkan's paper
    # dist_comp counter (theoretically, not practically, i.e. the implementation needs more than counted)
    dist_comp_pto = 0
    # ptolemy
    dim = len(data[0])
    pivot = np.zeros(dim)
    pivot_dist_data = np.linalg.norm(data - pivot, axis=1)
    dist_comp_pto += n
    # to be compatible with the code between 4 and 5; we need "..._new_center" for the first "..._old_center"
    pivot_dist_new_center = np.linalg.norm(centroids - pivot, axis=1)
    dist_comp_pto += k

    # numbers as in Elkan's original paper
    while not converged:
        iteration_count += 1
        if iteration_count > 1:
            converged = True

        # 1
        center_distances = np.array([np.linalg.norm(centroids - centroids[i], axis=1) for i in range(k)])
        center_distances[center_distances == 0] = [np.nan]
        center_bounds = 1 / 2 * np.nanmin(center_distances, axis=0)
        dist_comp_pto += k * (k-1) / 2

        # 2
        center_condition = center_bounds[assignment] < upper_bounds  # atm list of booleans
        relevant_indices = indices[center_condition]
        relevant_assignment = assignment[center_condition]
        relevant_upper_bound = upper_bounds[center_condition]
        # saved_dist_comp += n - len(relevant_indices) * ...
        # tri und pto unterscheiden sich ^hier^, daher ist es nicht einfach beides innerhalb einer Methode zu zählen

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
            center_condition_again = 1 / 2 * center_distances[center_index, relevant_assignment] < relevant_upper_bound

            all_conditions = not_same_center & lu_bound_condition & center_condition_again
            unpruned_indices = relevant_indices[all_conditions]

            # 3a - d(x,c(x))
            r_condition = loose_upper_bound[unpruned_indices]
            need_to_update_indices = unpruned_indices[r_condition]
            nooo_need_to_update_indices = unpruned_indices[np.logical_not(r_condition)]
            # compute distances
            distances[need_to_update_indices, assignment[need_to_update_indices]] = \
                np.linalg.norm(data[need_to_update_indices] - centroids[assignment[need_to_update_indices]], axis=1)
            dist_comp_pto += len(need_to_update_indices)
            # update bounds after distance computation
            upper_bounds[need_to_update_indices] = distances[need_to_update_indices, assignment[need_to_update_indices]]
            lower_bounds[need_to_update_indices, assignment[need_to_update_indices]] = \
                distances[need_to_update_indices, assignment[need_to_update_indices]]
            # distance equals upper bound for these indices now
            loose_upper_bound[need_to_update_indices] = False

            # if loose_upper_bound is False: upper bound is still tight
            distances[nooo_need_to_update_indices, assignment[nooo_need_to_update_indices]] \
                = upper_bounds[nooo_need_to_update_indices]

            # 3b - d(x,c)
            condition_3b = distances[unpruned_indices, assignment[unpruned_indices]] > np.min(
                [lower_bounds[unpruned_indices, center_index],
                 1 / 2 * center_distances[assignment[unpruned_indices], center_index]], axis=0)
            condition_3b_indices = unpruned_indices[condition_3b]
            # compute distances
            distances[condition_3b_indices, center_index] = np.linalg.norm(data[condition_3b_indices] - center, axis=1)
            dist_comp_pto += len(condition_3b_indices)
            # check, which points need a new assignment
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
        dist_comp_pto += k

        # ptolemy
        pivot_dist_old_center = pivot_dist_new_center.copy()
        pivot_dist_new_center = np.linalg.norm(new_centroids - pivot, axis=1)
        dist_comp_pto += k

        # 5
        for center_index, center in enumerate(centroids):
            lower_bounds[indices, center_index] = \
                (lower_bounds[indices, center_index] * pivot_dist_new_center[center_index] -
                 pivot_dist_data[indices] * centroids_movement[center_index]) / pivot_dist_old_center[center_index]

        # 6
        upper_bounds[indices] = ((upper_bounds[indices] * pivot_dist_new_center[assignment] +
                                  pivot_dist_data[indices] * centroids_movement[assignment]) /
                                 pivot_dist_old_center[assignment])
        loose_upper_bound[indices] = True

        # 7
        centroids = new_centroids

    print(f'Elkan Pto Dist_Comp {dist_comp_pto}')
    return centroids, assignment









