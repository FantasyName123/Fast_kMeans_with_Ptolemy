import matplotlib.pyplot as plt
import time

from kMeansVariants import *
from datasets import *
from subroutines import initialise_centroids


if __name__ == '__main__':
    # Todo: Skripte und grafische Auswertungen erstellen
    # bisher habe ich nur die oberen Schranken mit Ptolemy erstellt

    k = 10
    data = create_clustered_data(6000, dimension=6, clusters=k)

    initial_centroids = initialise_centroids(k=k, data=data)
    # todo: Die Berechnung des jeweiligen Zielfunktionswertes geht aktuell mit in die Zeitberechnung ein
    start = time.time()
    centroids, assignment, iterations_lloyd, assignment_1_lloyd = \
        lloyd_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_lloyd = calculate_zielfunktion(centroids, assignment) / len(assignment)

    step = time.time()
    centroids, assignment, iterations_elkan, saved_dist_comp_theory, saved_dist_comp_practice =\
        elkan_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_elkan = calculate_zielfunktion(centroids, assignment) / len(assignment)

    step2 = time.time()
    centroids, assignment, iterations_hamerly_pto, saved_dist_comp_theory_pto, saved_dist_comp_practice_pto =\
        hamerly_both_ptolemy_upper_bound_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_hamerly_pto = calculate_zielfunktion(centroids, assignment) / len(assignment)

    step3 = time.time()
    centroids, assignment, iterations_hybrid, saved_dist_comp_theory_hybrid, saved_dist_comp_practice_hybrid =\
        elkan_lower_ptolemy_upper_bound_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_hybrid = calculate_zielfunktion(centroids, assignment) / len(assignment)
    end = time.time()

    print(f'Lloyd Algorithm needed {round(step - start, 4)} seconds')
    print(f'Elkan Algorithm needed {round(step2 - step, 4)} seconds')
    print(f'Hamerly_Pto Algorithm needed {round(step3 - step2, 4)} seconds')
    print(f'Hybrid Algorithm needed {round(end - step3, 4)} seconds')

    print('Saved distance computations with Elkan:')
    print(f'In Theory: {saved_dist_comp_theory}')
    print(f'In Practice: {saved_dist_comp_practice}')
    print('Saved distance computations with Hamerly_Pto:')
    print(f'In Theory: {saved_dist_comp_theory_pto}')
    print(f'In Practice: {saved_dist_comp_practice_pto}')
    print('Saved distance computations with Hybrid:')
    print(f'In Theory: {saved_dist_comp_theory_hybrid}')
    print(f'In Practice: {saved_dist_comp_practice_hybrid}')

    # Zielfunktionswert
    if zielfunktionswert_elkan != zielfunktionswert_hybrid:
        print('Warning! The zielfunktionswerte were different...')
    print(f'Lloyd: {zielfunktionswert_lloyd}')
    print(f'Elkan: {zielfunktionswert_elkan}')
    print(f'Hamerly_Pto: {zielfunktionswert_hamerly_pto}')
    print(f'Hybrid: {zielfunktionswert_hybrid}')
    print(f'Iterations Lloyd: {iterations_lloyd}')
    print(f'Iterations Hamelry: {iterations_elkan}')
    print(f'Iterations Hamerly_Pto: {iterations_hamerly_pto}')
    print(f'Iterations Hybrid: {iterations_hybrid}')

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

