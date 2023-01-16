import numpy as np
from math import dist

from subroutines import *
from helpers import *


def lloyd_algorithm(data, k, initial_centroids):
    # initialise
    centroids = initial_centroids
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

        # update_center
        update_centroids(centroids, assignment)

    return centroids, assignment, iteration_count, 0, 0


def hamerly_algorithm(data, k, initial_centroids):
    # initialise
    centroids = initial_centroids
    assignment = dict.fromkeys(data, -1)
    upper_bounds_point = dict()
    lower_bounds_point = dict()
    center_bound = dict()
    saved_dist_comp_theory = 0
    saved_dist_comp_practice = 0

    # die bounds müssen erst einmal initialsiert werden, damit man in den ersten Durchlauf starten kann
    for point in data:
        # point_all_centers
        distances = all_dist(dataset=centroids, query=point)
        new_label = distances.argmin()
        assignment[point] = new_label
        upper_bounds_point[point] = distances.min()
        remaining_distances = distances.drop(new_label)
        lower_bounds_point[point] = remaining_distances.min()
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

    iteration_count = 0
    assignment_updated = True
    while assignment_updated:
        iteration_count += 1
        assignment_updated = False

        # -----------------------  Update Assignment -----------------------
        #   update center to center bounds
        for label, centroid in enumerate(centroids):
            saved_dist_comp_theory -= k * (k-1) / 2  # alle Werte oberhalb der Diagonale in der Abstandsmatrix
            saved_dist_comp_practice -= k * k
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            all_dist_centroid = all_dist_centroid.drop(label)
            new_bound = all_dist_centroid.min()
            center_bound[label] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # die lower bound ist zu hoch (manchmal, wenn man nur moved_distance[r_prime] abzieht)
            m = max(center_bound[label], lower_bounds_point[point])
            if upper_bounds_point[point] >= m:
                upper_bounds_point[point] = dist(centroids[label], point)
                if upper_bounds_point[point] >= m:
                    # point_all_centers
                    # hier wird aktuell eine Distanz berechnet, die wir schon kennen: dist(centroid[label], point)
                    # dadurch haben wir am Ende sogar eine Distanzberechnung mehr als bei Lloyd
                    # todo: dies optimieren, indem man diese Distanz nicht berechnen muss
                    distances = all_dist(dataset=centroids, query=point)
                    new_label = distances.argmin()
                    if assignment[point] != new_label:
                        assignment[point] = new_label
                        assignment_updated = True
                    upper_bounds_point[point] = distances.min()
                    remaining_distances = distances.drop(new_label)
                    lower_bounds_point[point] = remaining_distances.min()
                    saved_dist_comp_practice -= 1
                else:
                    # durch das Updaten der upper bound haben wir eine Distanzberechnung weniger gespart
                    saved_dist_comp_theory += k - 1
                    saved_dist_comp_practice += k - 1
            else:
                saved_dist_comp_theory += k
                saved_dist_comp_practice += k

        # update_center
        moved_distance = update_centroids(centroids, assignment)
        # k moved distances
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k
        if sum(moved_distance) != 0:
            assignment_updated = True

        # update_bounds
        r = moved_distance.argmax()
        remaining_moved_distances = moved_distance.drop(r)
        r_prime = remaining_moved_distances.argmax()
        for point in data:
            label = assignment[point]
            upper_bounds_point[point] += moved_distance[label]

            # todo: correct this problem:
            # in here the mistake is hidden. Sometimes it is not save to subtract only the second highest
            # moved distance. Is the label not correct??
            if r == label:
                pass
            if False:
                lower_bounds_point[point] -= moved_distance[r_prime]
            else:
                lower_bounds_point[point] -= moved_distance[r]

    return centroids, assignment, iteration_count, int(saved_dist_comp_theory), saved_dist_comp_practice


# todo (maybe): catch pivot_1 = pivot_2 errors (division by zero)
def hamerly_both_ptolemy_upper_bound_algorithm(data, k, initial_centroids):
    """
    This variant uses both triangle and Ptolemy inequality for the upper bound

    :param data:
    :param k:
    :param initial_centroids:
    :return:
    """
    # initialise
    centroids = initial_centroids
    assignment = dict.fromkeys(data, -1)
    upper_bounds_point = dict()
    lower_bounds_point = dict()
    center_bound = dict()
    saved_dist_comp_theory = 0
    saved_dist_comp_practice = 0

    # new
    moved_distance = [0] * k
    ptolemy_wins = 0
    triangle_wins = 0
    both_win = 0
    no_winner = 0

    # new (point_to_center distances)
    dimension = len(data[0])
    zero_1 = tuple([0] * 0 + [100] * (dimension - 0))
    zero_2 = tuple([0 for dim in range(dimension)])
    point_to_zero_1 = update_points_to_pivot_distances(data, zero_1, saved_dist_comp_theory, saved_dist_comp_practice)
    point_to_zero_2 = update_points_to_pivot_distances(data, zero_2, saved_dist_comp_theory, saved_dist_comp_practice)

    # die bounds müssen erst einmal initialsiert werden, damit man in den ersten Durchlauf starten kann
    for point in data:
        # point_all_centers
        distances = all_dist(dataset=centroids, query=point)
        new_label = distances.argmin()
        assignment[point] = new_label
        upper_bounds_point[point] = distances.min()
        remaining_distances = distances.drop(new_label)
        lower_bounds_point[point] = remaining_distances.min()
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

    # new (old_center_to_zeros distance
    # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
    old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)
    old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)

    iteration_count = 0
    assignment_updated = True
    while assignment_updated:
        iteration_count += 1
        assignment_updated = False

        # new (new_center_to_zero distances)
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        new_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        new_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # -----------------------  Update Assignment -----------------------
        #   update center to center bounds
        for label, centroid in enumerate(centroids):
            saved_dist_comp_theory -= k * (k-1) / 2  # alle Werte oberhalb der Diagonale in der Abstandsmatrix
            saved_dist_comp_practice -= k * k
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            all_dist_centroid = all_dist_centroid.drop(label)
            new_bound = all_dist_centroid.min()
            center_bound[label] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # die lower bound ist zu hoch (manchmal, wenn man nur moved_distance[r_prime] abzieht)
            m = max(center_bound[label], lower_bounds_point[point])
            upper_bound_ptolemy_1 = (dist(point, centroids[label]) * new_centers_to_zero_1[label]
                                     + point_to_zero_1[point] * moved_distance[label]) \
                                     / old_centers_to_zero_1[label]
            upper_bound_ptolemy_2 = (dist(point, centroids[label]) * new_centers_to_zero_2[label]
                                     + point_to_zero_2[point] * moved_distance[label]) \
                                     / old_centers_to_zero_2[label]
            upper_bound_ptolemy = min(upper_bound_ptolemy_1, upper_bound_ptolemy_2)
            saved_dist_comp_practice -= 2
            # Test which bound is better
            if upper_bounds_point[point] < m <= upper_bound_ptolemy:
                triangle_wins += 1
            elif upper_bound_ptolemy < m <= upper_bounds_point[point]:
                ptolemy_wins += 1
            elif max(upper_bounds_point[point], upper_bound_ptolemy) < m:
                both_win += 1
            else:
                no_winner += 1
            if upper_bounds_point[point] >= m or upper_bound_ptolemy >= m:
                upper_bounds_point[point] = dist(centroids[label], point)
                if upper_bounds_point[point] >= m or upper_bound_ptolemy >= m:
                    # point_all_centers
                    # hier wird aktuell eine Distanz berechnet, die wir schon kennen: dist(centroid[label], point)
                    # dadurch haben wir am Ende sogar eine Distanzberechnung mehr als bei Lloyd
                    # todo: dies optimieren, indem man diese Distanz nicht berechnen muss
                    distances = all_dist(dataset=centroids, query=point)
                    new_label = distances.argmin()
                    if assignment[point] != new_label:
                        assignment[point] = new_label
                        assignment_updated = True
                    upper_bounds_point[point] = distances.min()
                    remaining_distances = distances.drop(new_label)
                    lower_bounds_point[point] = remaining_distances.min()
                    saved_dist_comp_practice -= 1
                else:
                    # durch das Updaten der upper bound haben wir eine Distanzberechnung weniger gespart
                    saved_dist_comp_theory += k - 1
                    saved_dist_comp_practice += k - 1
            else:
                saved_dist_comp_theory += k
                saved_dist_comp_practice += k

        # new (old_center_to_zero distances)
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # update_center
        moved_distance = update_centroids(centroids, assignment)
        # k moved distances
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k
        if sum(moved_distance) != 0:
            assignment_updated = True

        # update_bounds
        r = moved_distance.argmax()
        remaining_moved_distances = moved_distance.drop(r)
        r_prime = remaining_moved_distances.argmax()
        for point in data:
            label = assignment[point]
            upper_bounds_point[point] += moved_distance[label]

            # todo: correct this problem:
            # inn here the mistake is hidden. Sometimes it is not save to subtract only the second highest
            # moved distance. Is the label not correct??
            if r == label:
                pass
            if False:
                lower_bounds_point[point] -= moved_distance[r_prime]
            else:
                lower_bounds_point[point] -= moved_distance[r]

    print(f'Triangle wins: {triangle_wins}')
    print(f'Ptolemy wins: {ptolemy_wins}')
    print(f'Both win: {both_win}')
    print(f'No winner: {no_winner}')
    return centroids, assignment, iteration_count, int(saved_dist_comp_theory), saved_dist_comp_practice


def hamerly_lower_ptolemy_upper_bound_algorithm(data, k, initial_centroids):
    """
    This variant replaces the usual upper bound used by Hamerly with an upper bound created with the Ptolemy inequality

    :param data:
    :param k:
    :param initial_centroids:
    :return:
    """
    # initialise
    centroids = initial_centroids
    assignment = dict.fromkeys(data, -1)
    lower_bounds_point = dict()
    center_bound = dict()
    saved_dist_comp_theory = 0
    saved_dist_comp_practice = 0

    # new
    moved_distance = [0] * k

    # new (point_to_center distances)
    dimension = len(data[0])
    zero_1 = tuple([0] * 0 + [100] * (dimension - 0))
    zero_2 = tuple([0 for dim in range(dimension)])
    point_to_zero_1 = update_points_to_pivot_distances(data, zero_1, saved_dist_comp_theory, saved_dist_comp_practice)
    point_to_zero_2 = update_points_to_pivot_distances(data, zero_2, saved_dist_comp_theory, saved_dist_comp_practice)

    # die bounds müssen erst einmal initialsiert werden, damit man in den ersten Durchlauf starten kann
    for point in data:
        # point_all_centers
        distances = all_dist(dataset=centroids, query=point)
        new_label = distances.argmin()
        assignment[point] = new_label
        remaining_distances = distances.drop(new_label)
        lower_bounds_point[point] = remaining_distances.min()
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

    # new (old_center_to_zeros distance)
    # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
    old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)
    old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)

    iteration_count = 0
    assignment_updated = True
    while assignment_updated:
        iteration_count += 1
        assignment_updated = False

        # new (new_center_to_zero distances)
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        new_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        new_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # -----------------------  Update Assignment -----------------------
        #   update center to center bounds
        for label, centroid in enumerate(centroids):
            saved_dist_comp_theory -= k * (k-1) / 2  # alle Werte oberhalb der Diagonale in der Abstandsmatrix
            saved_dist_comp_practice -= k * k
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            all_dist_centroid = all_dist_centroid.drop(label)
            new_bound = all_dist_centroid.min()
            center_bound[label] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # die lower bound ist zu hoch (manchmal, wenn man nur moved_distance[r_prime] abzieht)
            m = max(center_bound[label], lower_bounds_point[point])
            upper_bound_ptolemy_1 = (dist(point, centroids[label]) * new_centers_to_zero_1[label]
                                     + point_to_zero_1[point] * moved_distance[label]) \
                                     / old_centers_to_zero_1[label]
            upper_bound_ptolemy_2 = (dist(point, centroids[label]) * new_centers_to_zero_2[label]
                                     + point_to_zero_2[point] * moved_distance[label]) \
                                     / old_centers_to_zero_2[label]
            upper_bound_ptolemy = min(upper_bound_ptolemy_1, upper_bound_ptolemy_2)
            saved_dist_comp_practice -= 2
            if upper_bound_ptolemy >= m:
                # point_all_centers
                # hier wird aktuell eine Distanz berechnet, die wir schon kennen: dist(centroid[label], point)
                # dadurch haben wir am Ende sogar eine Distanzberechnung mehr als bei Lloyd
                # todo: dies optimieren, indem man diese Distanz nicht berechnen muss
                distances = all_dist(dataset=centroids, query=point)
                new_label = distances.argmin()
                if assignment[point] != new_label:
                    assignment[point] = new_label
                    assignment_updated = True
                remaining_distances = distances.drop(new_label)
                lower_bounds_point[point] = remaining_distances.min()
                saved_dist_comp_practice -= 1
            else:
                saved_dist_comp_theory += k
                saved_dist_comp_practice += k

        # new (old_center_to_zero distances)
        all_dist_old_centers_zero_1 = all_dist(centroids, query=zero_1)
        all_dist_old_centers_zero_2 = all_dist(centroids, query=zero_2)
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # update_center
        moved_distance = update_centroids(centroids, assignment)
        # k moved distances
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k
        if sum(moved_distance) != 0:
            assignment_updated = True

        # update_bounds
        r = moved_distance.argmax()
        remaining_moved_distances = moved_distance.drop(r)
        r_prime = remaining_moved_distances.argmax()
        for point in data:
            label = assignment[point]
            # todo: correct this problem:
            # inn here the mistake is hidden. Sometimes it is not save to subtract only the second highest
            # moved distance. Is the label not correct??
            if r == label:
                pass
            if False:
                lower_bounds_point[point] -= moved_distance[r_prime]
            else:
                lower_bounds_point[point] -= moved_distance[r]

    return centroids, assignment, iteration_count, int(saved_dist_comp_theory), saved_dist_comp_practice


def elkan_algorithm(data, k, initial_centroids):
    # Initialisation
    centroids = initial_centroids
    assignment = dict()
    # the points are the keys and each value is a list of lower bounds with length k
    lower_bounds = dict()
    upper_bound = dict()
    r = dict.fromkeys(data, True)
    center_bound = dict()
    center_center_dist = dict()
    saved_dist_comp_theory = 0
    saved_dist_comp_practice = 0

    for point in data:
        all_dist_this_point = all_dist(dataset=centroids, query=point)
        new_label = all_dist_this_point.argmin()
        assignment[point] = new_label
        distances = list(all_dist_this_point)
        lower_bounds[point] = distances
        min_dist = all_dist_this_point[new_label]
        upper_bound[point] = min_dist
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

    iteration_count = 0
    not_converged = True
    while not_converged:
        iteration_count += 1
        # in the first iteration only the centroids get updated, and we do not visit the convergence condition
        if iteration_count > 2:
            not_converged = False
        # 1
        for label, centroid in enumerate(centroids):
            saved_dist_comp_theory -= k * (k-1) / 2  # alle Werte oberhalb der Diagonale in der Abstandsmatrix
            saved_dist_comp_practice -= k * k
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            for second_label in all_dist_centroid.index:
                center_center_dist[(label, second_label)] = all_dist_centroid[second_label]
            all_dist_centroid = all_dist_centroid.drop(label)
            new_bound = all_dist_centroid.min()
            center_bound[label] = 1/2 * new_bound

        for point in data:
            label = assignment[point]
            # 2
            if upper_bound[point] <= center_bound[label]:
                saved_dist_comp_theory += k
                saved_dist_comp_practice += k
                pass
            # 3
            else:
                for other_label in [lab for lab in range(k) if lab != label]:
                    if upper_bound[point] > lower_bounds[point][other_label] and \
                            upper_bound[point] > 1/2 * center_center_dist[(label, other_label)]:
                        # 3a
                        if r[point]:
                            distance = dist(point, centroids[label])
                            upper_bound[point] = distance
                            lower_bounds[point][label] = distance
                            r[point] = False
                        else:
                            distance = upper_bound[point]

                            # saved_dist_comp so richtig berechnet ???
                            saved_dist_comp_theory += 1
                            saved_dist_comp_practice += 1
                            if distance != dist(point, centroids[label]):
                                print('Housten, we have a problem')
                                raise RuntimeError('Upper bound was not equal to distance')

                        # 3b
                        if distance > lower_bounds[point][other_label] or \
                                distance > 1/2 * center_center_dist[(label, other_label)]:
                            distance_other_center = dist(point, centroids[other_label])
                            lower_bounds[point][other_label] = distance_other_center
                            if distance_other_center < distance:
                                assignment[point] = other_label
                                label = other_label  # !!!!!!!!
                                upper_bound[point] = distance_other_center
                                not_converged = True
                        else:
                            # saved_dist_comp so richtig berechnet ???
                            saved_dist_comp_theory += 1
                            saved_dist_comp_practice += 1

        # 4 and 7
        moved_distance = update_centroids(centroids, assignment)
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

        for point in data:
            # 5
            for label in range(k):
                lower_bounds[point][label] = max(0, lower_bounds[point][label] - moved_distance[label])
            # 6
            upper_bound[point] += moved_distance[assignment[point]]
            r[point] = True

    return centroids, assignment, iteration_count, int(saved_dist_comp_theory), saved_dist_comp_practice


# todo: das mal vernünftig machen, mit der upper_bound_ptolemy: Wann die berechnet wird, wie die mitgenommen wird usw.
def elkan_lower_ptolemy_upper_bound_algorithm(data, k, initial_centroids):
    """

    :param data:
    :param k:
    :param initial_centroids:
    :return:
    """
    # Initialisation
    centroids = initial_centroids
    assignment = dict()
    upper_bound_ptolemy = dict()
    # the points are the keys and each value is a list of lower bounds with length k
    lower_bounds = dict()
    r = dict.fromkeys(data, True)
    center_bound = dict()
    center_center_dist = dict()
    saved_dist_comp_theory = 0
    saved_dist_comp_practice = 0

    # new
    moved_distance = [0] * k

    # new (point_to_center distances)
    dimension = len(data[0])
    zero_1 = tuple([0] * 0 + [100] * (dimension - 0))
    zero_2 = tuple([0 for dim in range(dimension)])
    point_to_zero_1 = update_points_to_pivot_distances(data, zero_1, saved_dist_comp_theory,
                                                       saved_dist_comp_practice)
    point_to_zero_2 = update_points_to_pivot_distances(data, zero_2, saved_dist_comp_theory,
                                                       saved_dist_comp_practice)

    # Initialise bounds
    for point in data:
        all_dist_this_point = all_dist(dataset=centroids, query=point)
        new_label = all_dist_this_point.argmin()
        assignment[point] = new_label
        distances = list(all_dist_this_point)
        lower_bounds[point] = distances
        min_dist = all_dist_this_point[new_label]
        upper_bound_ptolemy[point] = min_dist
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

    # new (old_center_to_zeros distance)
    # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
    old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)
    old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                              saved_dist_comp_practice)

    iteration_count = 0
    not_converged = True
    while not_converged:
        iteration_count += 1
        # in the first iteration only the centroids get updated, and we do not visit the convergence condition
        if iteration_count > 2:
            not_converged = False

        # new (new_center_to_zero distances)
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        new_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        new_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # 1
        for label, centroid in enumerate(centroids):
            saved_dist_comp_theory -= k * (k - 1) / 2  # alle Werte oberhalb der Diagonale in der Abstandsmatrix
            saved_dist_comp_practice -= k * k
            all_dist_centroid = all_dist(dataset=centroids, query=centroid)
            for second_label in all_dist_centroid.index:
                center_center_dist[(label, second_label)] = all_dist_centroid[second_label]
            all_dist_centroid = all_dist_centroid.drop(label)
            new_bound = all_dist_centroid.min()
            center_bound[label] = 1 / 2 * new_bound

        for point in data:
            label = assignment[point]
            # 2
            # ptolemy upper bound
            # upper_bound_ptolemy_1 = (dist(point, centroids[label]) * new_centers_to_zero_1[label]
            #                          + point_to_zero_1[point] * moved_distance[label]) \
            #                          / old_centers_to_zero_1[label]
            # upper_bound_ptolemy_2 = (dist(point, centroids[label]) * new_centers_to_zero_2[label]
            #                          + point_to_zero_2[point] * moved_distance[label]) \
            #                          / old_centers_to_zero_2[label]
            # saved_dist_comp_practice -= 2
            upper_bound_ptolemy_1 = (upper_bound_ptolemy[point] * new_centers_to_zero_1[label]
                                     + point_to_zero_1[point] * moved_distance[label]) \
                                     / old_centers_to_zero_1[label]
            upper_bound_ptolemy_2 = (upper_bound_ptolemy[point] * new_centers_to_zero_2[label]
                                     + point_to_zero_2[point] * moved_distance[label]) \
                                     / old_centers_to_zero_2[label]
            upper_bound_ptolemy_int = min(upper_bound_ptolemy_1, upper_bound_ptolemy_2)
            upper_bound_ptolemy[point] = upper_bound_ptolemy_int
            if upper_bound_ptolemy_int <= center_bound[label]:
                saved_dist_comp_theory += k
                saved_dist_comp_practice += k
                pass
            # 3
            else:
                for other_label in [lab for lab in range(k) if lab != label]:
                    if upper_bound_ptolemy_int > lower_bounds[point][other_label] and \
                            upper_bound_ptolemy_int > 1 / 2 * center_center_dist[(label, other_label)]:
                        # 3a
                        if r[point]:
                            saved_dist_comp_theory -= 1
                            saved_dist_comp_practice -= 1
                            distance = dist(point, centroids[label])
                            # die ptolomäische Variante profitiert noch nicht davon, dass man die upper bound hier
                            # aktualisieren könnte
                            # upper_bound[point] = distance # ???
                            lower_bounds[point][label] = distance
                            # todo: mit der folgenden auskommentierten Zeile hat es nicht funktioniert. Wieso?
                            # r[point] = False

                        # 3b
                        if distance > lower_bounds[point][other_label] or \
                                distance > 1 / 2 * center_center_dist[(label, other_label)]:
                            saved_dist_comp_theory -= 1
                            saved_dist_comp_practice -= 1
                            distance_other_center = dist(point, centroids[other_label])
                            lower_bounds[point][other_label] = distance_other_center
                            if distance_other_center < distance:
                                assignment[point] = other_label
                                label = other_label  # !!!!!!!!
                                # upper_bound[point] = distance_other_center # ???
                                not_converged = True

        # new (old_center_to_zero distances)
        # wir indizieren hier nicht mit den Zentren, sondern mit den Labels.
        old_centers_to_zero_1 = update_centers_to_pivot_distances(centroids, zero_1, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)
        old_centers_to_zero_2 = update_centers_to_pivot_distances(centroids, zero_2, saved_dist_comp_theory,
                                                                  saved_dist_comp_practice)

        # 4 and 7 (update center)
        moved_distance = update_centroids(centroids, assignment)
        saved_dist_comp_theory -= k
        saved_dist_comp_practice -= k

        for point in data:
            # 5
            for label in range(k):
                lower_bounds[point][label] = max(0, lower_bounds[point][label] - moved_distance[label])
            # 6
            r[point] = True

    return centroids, assignment, iteration_count, int(saved_dist_comp_theory), saved_dist_comp_practice


