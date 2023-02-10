import matplotlib.pyplot as plt
import time
import numpy as np

from scripts import *
from kMeansVariants import *
from kMeansVariants_matrix import *
from datasets import *
from subroutines import initialise_centroids

# todo: Elkan ist schneller als Hamerly, obwohl Elkan weniger Distanzberechnung einspart. Das sollte eigentlich nicht
#  sein, denn der Overhead bei Elkan ist größer. Untersuchen!
# Eine Idee habe ich schon:
#   Manchmal werden Distanzen vektorisiert berechnet und manchmal einzelne in einer Schleife. Diese beiden Arten von
# Distanzberechnungen sind natürlich nicht miteinander vergleichbar. Das kann dazu führen, dass ein Algorithmus mit mehr
# Distanzberechnungen schneller ist, weil er die meisten Distanzberechnungen vektorisiert durchführt.
#   Man könnte natürlich auch sagen, dass es eine positive Eigenschaft eines Algorithmus ist, wenn dieser viele
# vektorisierte Distanzberechnungen zulässt.
#
# Zum Teil wurden die saved_dist_comp auch falsch berechnet

# todo: die Anzahl der gesparten Distanzberechnungen war bei elkan_lower_ptolemy_upper_bound_algorithm_single teilweise
#  höher als bei elkan_style_triangle_ptolemy_combined_algorithm_single. Das sollte definitiv nicht passieren, denn:
#  Die beiden Algorithmen sind gleich bis darauf, dass letztere für die obere Schranke das Minimum der oberen Schranken
#  aus der Ptolomäischen- und der Dreiecksungleichung kombiniert und erstere nur die ptolomäische Schranke benutzt.

if __name__ == '__main__':
    # Todo: Skripte und grafische Auswertungen erstellen
    # bisher habe ich nur die oberen Schranken mit Ptolemy erstellt

    k = 100
    # data = create_clustered_data(40000, dimension=3, clusters=k)
    # data = create_dataset_uniform(5000, dimension=6)
    # data = get_birch_1()
    data = get_birch_2_1('/Users/louisborchert/Downloads/b2-random-txt/b2-random-60.txt')
    data_matrix = np.array(data)
    initial_centroids = initialise_centroids_from_dataset(data=data, k=k, seed=21)
    initial_centroids_matrix = np.array(initial_centroids)

    # # for testing purposes
    # data_matrix = get_test_dataset()
    # initial_centroids_matrix = np.array([(0, 0), (2, 2), (5, 5)])

    start = time.time()
    centroids, clusters = lloyd_matrix(data_matrix, np.copy(initial_centroids_matrix))
    step = time.time()
    centroids_test, clusters_test = elkan_matrix(data_matrix, np.copy(initial_centroids_matrix))
    end = time.time()

    print(f'Lloyd needed {round(step- start, 6)} seconds')
    print(f'Elkan needed {round(end - step, 6)} seconds')

    print(f'Centroids Test: {np.allclose(centroids, centroids_test)}')
    print(f'Clusters Test: {np.allclose(clusters, clusters_test)}')

    # # Visualisation matrix form
    # coordinates = data_matrix.transpose()
    # plt.scatter(x=coordinates[0], y=coordinates[1], c=list(clusters))
    # for centroid in centroids:
    #     plt.plot(centroid[0], centroid[1], marker='^')
    # plt.show()



    if True:
        # alg_list = [lloyd_algorithm, elkan_algorithm_without_counting]
        # names_list = ['Lloyd old', 'Elkan old']
        #
        # run(data, k, initial_centroids, alg_list, names_list)

        assignment_sklearn, centroids_sklearn = test_sklearn(data, k)

        # iterations_lloyd, zielfunktionswert_lloyd, runtime_lloyd, \
        #     saved_dist_comp_theory_lloyd, saved_dist_comp_practice_lloyd = \
        #     single_algorithm(lloyd_algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())

    if False:
        iterations_elkan, zielfunktionswert_elkan, runtime_elkan,\
            saved_dist_comp_theory_elkan, saved_dist_comp_practice_elkan = \
            single_algorithm(elkan_algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())
        print('Elkan done')

        iterations_elkan_pto, zielfunktionswert_elkan_pto, runtime_elkan_pto,\
            saved_dist_comp_theory_elkan_pto, saved_dist_comp_practice_elkan_pto = \
            single_algorithm(elkan_lower_ptolemy_upper_bound_algorithm_single, data=data, k=k,
                             initial_centroids=initial_centroids.copy())

        iterations_elkan_pto_2, zielfunktionswert_elkan_pto_2, runtime_elkan_pto_2,\
            saved_dist_comp_theory_elkan_pto_2, saved_dist_comp_practice_elkan_pto_2 = \
            single_algorithm(elkan_style_ptolemy_both_bounds_algorithm_single, data=data, k=k,
                             initial_centroids=initial_centroids.copy())

        # start = time.time()
        # centroids, assignment, iterations_lloyd, saved_dist_comp_theory_zero, saved_dist_comp_practice_zero = \
        #     lloyd_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
        # zielfunktionswert_lloyd = calculate_zielfunktion(centroids, assignment) / len(assignment)

        # print(f'Lloyd Algorithm needed {runtime_lloyd} seconds')
        print(f'Elkan Algorithm needed {runtime_elkan} seconds')
        print(f'Elkan_Ptolemy Algorithm Single needed {runtime_elkan_pto} seconds')
        print(f'Elkan_Ptolemy Algorithm Variant needed {runtime_elkan_pto_2} seconds')

        print('Saved distance computations with Elkan:')
        print(f'In Theory: {saved_dist_comp_theory_elkan}')
        print(f'In Practice: {saved_dist_comp_practice_elkan}')
        print('Saved distance computations with Elkan_Ptolemy Single:')
        print(f'In Theory: {saved_dist_comp_theory_elkan_pto}')
        print(f'In Practice: {saved_dist_comp_practice_elkan_pto}')
        print('Saved distance computations with Elkan_Ptolemy Variant:')
        print(f'In Theory: {saved_dist_comp_theory_elkan_pto_2}')
        print(f'In Practice: {saved_dist_comp_practice_elkan_pto_2}')

        # Zielfunktionswert
        # if zielfunktionswert_lloyd == zielfunktionswert_elkan == zielfunktionswert_elkan_pto == zielfunktionswert_elkan_pto_2:
        if zielfunktionswert_elkan == zielfunktionswert_elkan_pto == zielfunktionswert_elkan_pto_2:
            print(f'Zielfunktionswert: {zielfunktionswert_elkan}')
        else:
            print(' ------------------ Warning: Zielfunktionswerte were not equal!!! ------------------ ')

        if iterations_elkan == iterations_elkan_pto == iterations_elkan_pto_2:
            print(f'Iterations : {iterations_elkan}')
        else:
            print(' ------------------ Warning: Iterations were not equal!!! ------------------ ')

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

