from kMeansVariants import *
from subroutines import calculate_zielfunktion

import time


def single_algorithm(algorithm, data, k, initial_centroids):
    start = time.time()
    centroids, assignment, iterations, saved_dist_comp_theory, saved_dist_comp_practice = \
        algorithm(data, k, initial_centroids)
    end = time.time()
    runtime = round(end - start, 4)
    zielfunktionswert = calculate_zielfunktion(centroids, assignment) / len(assignment)

    return iterations, zielfunktionswert, runtime, saved_dist_comp_theory, saved_dist_comp_practice
