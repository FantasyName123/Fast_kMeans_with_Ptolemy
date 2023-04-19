import matplotlib.pyplot as plt
import time
import numpy as np

from scripts import *
from kMeansVariants_old import *
from kMeansVariants import *
from datasets import *
from subroutines import initialise_centroids

# Todo: Skripte und grafische Auswertungen erstellen

# todo: The elkan_matrix and elkan_matrix_full_pto need a different amount of dist_comp each time you run them
#  although the start configurations are the same

if __name__ == '__main__':
    show_visualisation = False
    compare_with_sklearn = True
    np.random.seed(42)

    k = 100
    data = get_birch_2_subset(size=22)
    # other possibilities to create a dataset:
    #   data = create_clustered_data(40000, dimension=3, clusters=k)
    #   data = create_dataset_uniform(5000, dimension=6)
    #   data = get_birch_1()
    data = np.array(data)
    initial_centroids = initialise_centroids_from_dataset(data=data, k=k, seed=0)
    initial_centroids_matrix = np.array(initial_centroids)

    start = time.time()
    centroids, clusters = lloyd(data, np.copy(initial_centroids_matrix))
    step = time.time()
    centroids_elkan, clusters_elkan = elkan_counting(data, np.copy(initial_centroids_matrix))
    step2 = time.time()
    centroids_elkan_pto, clusters_elkan_pto = elkan_full_ptolemy_counting(data, np.copy(initial_centroids_matrix))
    end = time.time()

    print(f'Lloyd needed {round(step- start, 6)} seconds')
    print(f'Elkan needed {round(step2 - step, 6)} seconds')
    print(f'Elkan Pto needed: {round(end - step2, 6)} seconds')

    print(f'Centroids Test: {np.array_equal(centroids, centroids_elkan)}')
    print(f'Clusters Test: {np.array_equal(clusters, clusters_elkan)}')
    print(f'Centroids Test Pto: {np.array_equal(centroids, centroids_elkan_pto)}')
    print(f'Clusters Test Pto: {np.array_equal(clusters, clusters_elkan_pto)}')

    # Visualisation
    if show_visualisation:
        coordinates = data.transpose()
        plt.scatter(x=coordinates[0], y=coordinates[1], c=list(clusters))
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], marker='^')
    plt.show()

    # Compare with sklearn's implementation
    if compare_with_sklearn:
        assignment_sklearn, centroids_sklearn = test_sklearn(data, k)

    # ------------------ Archive ---------------------
    # iterations_elkan, objective_function_value_elkan, runtime_elkan,\
    #     saved_dist_comp_theory_elkan, saved_dist_comp_practice_elkan = \
    #     single_algorithm(elkan_algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())
    # print('Elkan done')
    #
    # iterations_elkan_pto, objective_function_value_elkan_pto, runtime_elkan_pto,\
    #     saved_dist_comp_theory_elkan_pto, saved_dist_comp_practice_elkan_pto = \
    #     single_algorithm(elkan_lower_ptolemy_upper_bound_algorithm_single, data=data, k=k,
    #                      initial_centroids=initial_centroids.copy())
    #
    # iterations_elkan_pto_2, objective_function_value_elkan_pto_2, runtime_elkan_pto_2,\
    #     saved_dist_comp_theory_elkan_pto_2, saved_dist_comp_practice_elkan_pto_2 = \
    #     single_algorithm(elkan_style_ptolemy_both_bounds_algorithm_single, data=data, k=k,
    #                      initial_centroids=initial_centroids.copy())
    #
    # # start = time.time()
    # # centroids, assignment, iterations_lloyd, saved_dist_comp_theory_zero, saved_dist_comp_practice_zero = \
    # #     lloyd_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    # # objective_function_value_lloyd = calculate_zielfunktion(centroids, assignment) / len(assignment)
    #
    # # print(f'Lloyd Algorithm needed {runtime_lloyd} seconds')
    # print(f'Elkan Algorithm needed {runtime_elkan} seconds')
    # print(f'Elkan_Ptolemy Algorithm Single needed {runtime_elkan_pto} seconds')
    # print(f'Elkan_Ptolemy Algorithm Variant needed {runtime_elkan_pto_2} seconds')
    #
    # print('Saved distance computations with Elkan:')
    # print(f'In Theory: {saved_dist_comp_theory_elkan}')
    # print(f'In Practice: {saved_dist_comp_practice_elkan}')
    # print('Saved distance computations with Elkan_Ptolemy Single:')
    # print(f'In Theory: {saved_dist_comp_theory_elkan_pto}')
    # print(f'In Practice: {saved_dist_comp_practice_elkan_pto}')
    # print('Saved distance computations with Elkan_Ptolemy Variant:')
    # print(f'In Theory: {saved_dist_comp_theory_elkan_pto_2}')
    # print(f'In Practice: {saved_dist_comp_practice_elkan_pto_2}')
    #
    # # objective_function_value
    # # if objective_function_value_lloyd == objective_function_value_elkan == objective_function_value_elkan_pto == objective_function_value_elkan_pto_2:
    # if objective_function_value_elkan == objective_function_value_elkan_pto == objective_function_value_elkan_pto_2:
    #     print(f'objective_function_value: {objective_function_value_elkan}')
    # else:
    #     print(' ------------------ Warning: objective_function_valuee were not equal!!! ------------------ ')
    #
    # if iterations_elkan == iterations_elkan_pto == iterations_elkan_pto_2:
    #     print(f'Iterations : {iterations_elkan}')
    # else:
    #     print(' ------------------ Warning: Iterations were not equal!!! ------------------ ')

    # # For Debugging
    # same = (assignment_1_lloyd == assignment_1_elkan)
    # first_difference = False
    # i = 0
    # while not first_difference:
    #     point = data[i]
    #     if assignment_1_lloyd[point] != assignment_1_elkan[point]:
    #         first_difference = point
    #     i += 1

    # # Visualisation
    # coordinates = np.array(list(assignment_sklearn.keys())).transpose()
    # plt.scatter(x=coordinates[0], y=coordinates[1], c=list(assignment_sklearn.values()))
    # for centroid in centroids_sklearn:
    #     plt.plot(centroid[0], centroid[1], marker='^')
    # plt.show()

