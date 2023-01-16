import matplotlib.pyplot as plt
import time

from scripts import *
from kMeansVariants import *
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


if __name__ == '__main__':
    # Todo: Skripte und grafische Auswertungen erstellen
    # bisher habe ich nur die oberen Schranken mit Ptolemy erstellt

    k = 10
    data = create_clustered_data(10000, dimension=16, clusters=k)

    initial_centroids = initialise_centroids(k=k, data=data)

    iterations_lloyd, zielfunktionswert_lloyd, runtime_lloyd, \
        saved_dist_comp_theory_lloyd, saved_dist_comp_practice_lloyd = \
        single_algorithm(lloyd_algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())

    iterations_elkan, zielfunktionswert_elkan, runtime_elkan,\
        saved_dist_comp_theory_elkan, saved_dist_comp_practice_elkan = \
        single_algorithm(elkan_algorithm, data=data, k=k, initial_centroids=initial_centroids.copy())

    iterations_elkan_pto, zielfunktionswert_elkan_pto, runtime_elkan_pto,\
        saved_dist_comp_theory_elkan_pto, saved_dist_comp_practice_elkan_pto = \
        single_algorithm(elkan_lower_ptolemy_upper_bound_algorithm, data=data, k=k,
                         initial_centroids=initial_centroids.copy())

    # start = time.time()
    # centroids, assignment, iterations_lloyd, saved_dist_comp_theory_zero, saved_dist_comp_practice_zero = \
    #     lloyd_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    # zielfunktionswert_lloyd = calculate_zielfunktion(centroids, assignment) / len(assignment)

    print(f'Lloyd Algorithm needed {runtime_lloyd} seconds')
    print(f'Elkan Algorithm needed {runtime_elkan} seconds')
    print(f'Elkan_Ptolemy Algorithm needed {runtime_elkan_pto} seconds')

    print('Saved distance computations with Elkan:')
    print(f'In Theory: {saved_dist_comp_theory_elkan}')
    print(f'In Practice: {saved_dist_comp_practice_elkan}')
    print('Saved distance computations with Elkan_Ptolemy:')
    print(f'In Theory: {saved_dist_comp_theory_elkan_pto}')
    print(f'In Practice: {saved_dist_comp_practice_elkan_pto}')

    # Zielfunktionswert
    print(f'Lloyd: {zielfunktionswert_lloyd}')
    print(f'Elkan: {zielfunktionswert_elkan}')
    print(f'Elkan Ptolemy: {zielfunktionswert_elkan_pto}')
    print(f'Iterations Lloyd: {iterations_lloyd}')
    print(f'Iterations Elkan: {iterations_elkan}')
    print(f'Iterations Elkan Ptolemy: {iterations_elkan_pto}')

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
    # coordinates = np.array(list(assignment.keys())).transpose()
    # plt.scatter(x=coordinates[0], y=coordinates[1], c=list(assignment.values()))
    # for centroid in centroids:
    #     plt.plot(centroid[0], centroid[1], marker='^')
    # plt.show()

