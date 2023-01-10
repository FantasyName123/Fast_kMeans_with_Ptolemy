import numpy as np
from math import dist

from subroutines import *
from helpers import *


def lloyd_algorithm(data, k, initial_centroids):
    # initialise
    centroids = initial_centroids
    print(f'Lloyd: {centroids}')
    assignment = dict.fromkeys(data, -1)

    iteration_count = 0
    assignment_updated = True
    while assignment_updated:
        iteration_count += 1

        # update_assignment
        assignment_updated = False
        for point in data:
            distances = all_dist(dataset=centroids, query=point)
            new_label = distances.argmin()
            if assignment[point] != new_label:
                assignment[point] = new_label
                assignment_updated = True

        if iteration_count == 2:
            assignment_1 = assignment.copy()

        # update_center
        update_centroids(centroids, assignment)

    return centroids, assignment, iteration_count, assignment_1


# todo: 1) Anzahl gesparter Distanzberechnungen ausgeben
def hamerly_algorithm(data, k, initial_centroids):
    # initialise
    centroids = initial_centroids
    print(f'Hamerly: {centroids}')
    assignment = dict.fromkeys(data, -1)
    upper_bounds_point = dict()
    lower_bounds_point = dict()
    center_bound = dict()
    saved_dist_comp = 0
    for point in data:
        # point_all_centers
        distances_test = all_dist(dataset=centroids, query=point)
        new_label_test = distances_test.argmin()
        assignment[point] = new_label_test
        upper_bounds_point[point] = distances_test.min()
        remaining_distances = distances_test.drop(new_label_test)
        lower_bounds_point[point] = remaining_distances.min()

    iteration_count = 0
    assignment_updated = True
    while assignment_updated:
        iteration_count += 1
        assignment_updated = False
        # update_assignment
        #   update center to center bounds
        for idx, centroid in enumerate(centroids):
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            all_dist_centroid = all_dist_centroid.drop(idx)
            new_bound = all_dist_centroid.min()
            center_bound[idx] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # die lower bound ist zu hoch
            m = max(center_bound[label], lower_bounds_point[point])
            m = lower_bounds_point[point]
            distances_test = all_dist(dataset=centroids, query=point)
            new_label_test = distances_test.argmin()
            if assignment[point] != new_label_test:
                if upper_bounds_point[point] < m:
                    print('---------------')
                    print(iteration_count)
                    print(point)
                    print(upper_bounds_point[point])
                    print(lower_bounds_point[point])

            m = center_bound[label]
            if True or upper_bounds_point[point] >= m:
                upper_bounds_point[point] = dist(centroids[label], point)
                if upper_bounds_point[point] >= m:
                    # point_all_centers
                    # hier wird aktuell eine Distanz berechnet, die wir schon kennen: dist(centroid[label], point)
                    # dadurch haben wir am Ende sogar eine Distanzberechnung mehr als bei Hamerly
                    # todo: dies optimieren, indem man diese Distanz nicht berechnen muss
                    distances_test = all_dist(dataset=centroids, query=point)
                    new_label_test = distances_test.argmin()
                    if assignment[point] != new_label_test:
                        assignment[point] = new_label_test
                        assignment_updated = True
                    upper_bounds_point[point] = distances_test.min()
                    remaining_distances = distances_test.drop(new_label_test)
                    lower_bounds_point[point] = remaining_distances.min()
                else:
                    saved_dist_comp += k - 1
            else:
                saved_dist_comp += k

        # update_center
        moved_distance = update_centroids(centroids, assignment)
        if sum(moved_distance) != 0:
            assignment_updated = True

        # update_bounds
        r = moved_distance.argmax()
        remaining_moved_distances = moved_distance.drop(r)
        r_prime = remaining_moved_distances.argmax()
        for point in data:
            label = assignment[point]
            upper_bounds_point[point] += moved_distance[label]

            # inn here the mistake is hidden. Sometimes it is not save to subtract only the second highest
            # moved distance. Is the label not correct??
            if r == label:
                pass
            if False:
                lower_bounds_point[point] -= moved_distance[r_prime]
            else:
                lower_bounds_point[point] -= moved_distance[r]

    return centroids, assignment, iteration_count, saved_dist_comp


def elkan_algorithm(data, k, initial_centroids):
    # Initialisation
    centroids = initial_centroids
    print(f'Elkan: {centroids}')
    assignment = dict()
    # the points are the keys and each value is a list of lower bounds with length k
    lower_bounds = dict()
    upper_bound = dict()
    r = dict.fromkeys(data, True)
    center_bound = dict()
    for point in data:
        all_dist_this_point = all_dist(dataset=centroids, query=point)
        new_label = all_dist_this_point.argmin()
        assignment[point] = new_label
        distances = list(all_dist_this_point)
        lower_bounds[point] = distances
        min_dist = all_dist_this_point[new_label]
        upper_bound[point] = min_dist

    iteration_count = 0
    not_converged = True
    while not_converged:
        iteration_count += 1
        # in the first iteration only the centroids get updated, and we do not visit the convergence condition
        if iteration_count > 2:
            not_converged = False
        # 1
        for idx, centroid in enumerate(centroids):
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            all_dist_centroid = all_dist_centroid.drop(idx)
            new_bound = all_dist_centroid.min()
            center_bound[idx] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # 2
            if upper_bound[point] <= center_bound[label]:
                pass
            # 3
            else:
                for other_label in [lab for lab in range(k) if lab != label]:
                    # todo: dist(c_f(x), c_i) precomputen und abrufen, statt neu zu berechnen
                    if upper_bound[point] > lower_bounds[point][other_label] and \
                            upper_bound[point] > 1/2 * dist(centroids[label], centroids[other_label]):
                        # 3a
                        if r[point]:
                            distance = dist(point, centroids[label])
                            lower_bounds[point][label] = distance
                            # todo: mit der folgenden auskommentierten Zeile hat es nicht funktioniert. Wieso?
                            # r[point] = False
                        else:
                            distance = upper_bound[point]
                            if distance != dist(point, centroids[label]):
                                raise RuntimeError('Upper bound was not equal to distance')

                        # 3b
                        if distance > lower_bounds[point][other_label] or \
                                distance > 1/2 * dist(centroids[label], centroids[other_label]):
                            distance_other_center = dist(point, centroids[other_label])
                            lower_bounds[point][other_label] = distance_other_center
                            if distance_other_center < distance:
                                assignment[point] = other_label
                                label = other_label  # !!!!!!!!
                                not_converged = True

        # 4 and 7
        moved_distance = update_centroids(centroids, assignment)
        for point in data:
            # 5
            for label in range(k):
                lower_bounds[point][label] = max(0, lower_bounds[point][label] - moved_distance[label])
            # 6
            upper_bound[point] += moved_distance[assignment[point]]
            r[point] = True

    return centroids, assignment, iteration_count








