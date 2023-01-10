import matplotlib.pyplot as plt
import time

from kMeansVariants import *
from datasets import *
from subroutines import initialise_centroids


if __name__ == '__main__':
    data = [
        (12, 23),
        (80, 80),
        (10, 10),
        (1, 40),
        (49, 20),
        (99, 94),
        (70, 40),
        (84, 94),
        (70, 92)

    ]

    k = 4
    data = create_clustered_data(4000, dimension=2, clusters=k)

    initial_centroids = initialise_centroids(k=k, data=data)
    intial_centroid_set = [(1, 1), (23, 90), (47, 14), (40, 68), (91, 34), (30, 2), (55, 95), (90, 3), (4, 71), (60, 8)]
    initial_centroids = intial_centroid_set[0:k]
    # TODO: keiner der Algorithmen ist deterministisch!! Alle produzieren in verschieden Durchläufen unterschiedliche
    #  Ergebnisse mit den gleichen Daten und initialen Zentren!!
    # todo: Die Berechnung des jeweiligen Zielfunktionswertes geht aktuell mit in die Zeitberechnung ein
    start = time.time()
    centroids, assignment, iterations_lloyd, assignment_1_lloyd = lloyd_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_lloyd = calculate_zielfunktion(centroids, assignment) / len(assignment)
    step = time.time()
    centroids, assignment, iterations_hamerly, saved_dist_comp =\
        hamerly_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_hamerly = calculate_zielfunktion(centroids, assignment) / len(assignment)
    step2 = time.time()
    centroids, assignment, iterations_elkan =\
        elkan_algorithm(data=data, k=k, initial_centroids=initial_centroids.copy())
    zielfunktionswert_elkan = calculate_zielfunktion(centroids, assignment) / len(assignment)
    end = time.time()
    print(f'Lloyd Algorithm needed {round(step - start, 4)} seconds')
    print(f'Hamerly Algorithm needed {round(step2 - step, 4)} seconds')
    print(f'Elkan Algorithm needed {round(end - step2, 4)} seconds')

    # Zielfunktionswert
    if zielfunktionswert_lloyd != zielfunktionswert_elkan:
        print('Warning! The zielfunktionswerte were different...')
    print(f'Lloyd: {zielfunktionswert_lloyd}')
    print(f'Hamerly: {zielfunktionswert_hamerly}')
    print(f'Elkan: {zielfunktionswert_elkan}')
    print(f'Iterations Lloyd: {iterations_lloyd}')
    print(f'Iterations Hamelry: {iterations_hamerly}')
    print(f'Iterations Elkan: {iterations_elkan}')

    # same = (assignment_1_lloyd == assignment_1_elkan)
    # first_difference = False
    # i = 0
    # while not first_difference:
    #     point = data[i]
    #     if assignment_1_lloyd[point] != assignment_1_elkan[point]:
    #         first_difference = point
    #     i += 1


    # Visualisation
    coordinates = np.array(list(assignment.keys())).transpose()
    plt.scatter(x=coordinates[0], y=coordinates[1], c=list(assignment.values()))
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], marker='^')
    plt.show()

