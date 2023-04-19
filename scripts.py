from kMeansVariants_old import *
from subroutines import evaluate_objective_function

import time

# todo: Methoden mit/ohne Docstring
#  single_algorithm: Ohne
#  run: Da muss ich nochmal ran

# todo: April!! Die Methode single_algorithm auf die neuen vektorisierten kMeans Algorithmen anwenden.


def single_algorithm(algorithm, data, k, initial_centroids):
    """
    This is kind of a wrapper around a function call of any algorithm in kMeansVariants old!!! It calls the given method
    and, in addition, stops the time the algorithm needed and computes the objective function relative to the amount of
    data points.

    :param algorithm:
    :param data:
    :param k:
    :param initial_centroids:
    :return:
    """
    start = time.time()
    centroids, assignment, iterations, saved_dist_comp_theory, saved_dist_comp_practice = \
        algorithm(data, k, initial_centroids)
    end = time.time()
    runtime = round(end - start, 4)
    zielfunktionswert = evaluate_objective_function(centroids, assignment) / len(assignment)

    return iterations, zielfunktionswert, runtime, saved_dist_comp_theory, saved_dist_comp_practice


def run(data, k, initial_centroids, alg_list, names_list=['Elkan', 'Ptolemy_Upper_Elkan_Lower', 'Ptolemy_Elkan_Variant']):
    """
    Start a single run for each algorithm in the algorithm_list. The runtime as well as the saved distance computations
    are printed.

    :param data: data
    :param k: k
    :param initial_centroids: initial centroids
    :param alg_list: a list of functions that give the same output as the Lloyd algorithm.
    :param names_list: a list of names for the algorithms the algorithm list. The lengths of both lists must match.
    :return: None. Could be changed to runtime and/or saved_dist_comp.
    """
    if len(alg_list) != len(names_list):
        raise ValueError('alg_list and names_list do not have the same length')
    iterations_dict = dict()
    zielfunktionswert_dict = dict()
    runtime_dict = dict()
    saved_dc_theory_dict = dict()
    saved_dc_practice_dict = dict()
    for idx, algorithm in enumerate(alg_list):
        name = names_list[idx]
        iterations, zielfunktionswert, runtime, saved_dc_theory, saved_dc_practice = \
            single_algorithm(algorithm=algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())

        # save the output in dictionaries
        iterations_dict[name] = iterations
        zielfunktionswert_dict[name] = zielfunktionswert
        runtime_dict[name] = runtime
        saved_dc_theory_dict[name] = saved_dc_theory
        saved_dc_practice_dict[name] = saved_dc_practice

    print('-------------------------------')
    for name in names_list:
        print(f'{name} needed {runtime_dict[name]} seconds')

    print('-------------------------------')
    for name in names_list:
        print(f'Saved distance computations with {name}:')
        print(f'In Theory:   {saved_dc_theory_dict[name]}')
        print(f'In Practice: {saved_dc_practice_dict[name]}')
        print(f'Difference: {saved_dc_theory_dict[name] - saved_dc_practice_dict[name]}')
