import numpy as np
import pandas as pd
import random

from helpers import *


def initialise_centroids(data, k):
    dimension = len(data[0])
    np.random.seed(0)
    centroids = np.random.rand(k, dimension)
    centroids *= 120  # die Daten liegen nicht ganz in einem 100^n Würfel
    centroids = [tuple(row) for row in centroids]
    return centroids


def update_centroids(centroids, assignment):
    labels = range(len(centroids))
    data = list(assignment.keys())
    moved_distance = list()
    for label in labels:
        assigned_points = [point for point in data if assignment.get(point) == label]
        if len(assigned_points) == 0:
            # new_centroid = data[random.randint(0, len(data) - 1)]
            new_centroid = data[0]
            moved_distance.append(dist(centroids[label], new_centroid))
            centroids[label] = new_centroid
            continue
        arr = np.array(assigned_points)
        new_centroid = arr.sum(axis=0) / len(assigned_points)
        moved_distance.append(dist(centroids[label], new_centroid))
        centroids[label] = new_centroid

    moved_distance = pd.Series(moved_distance)
    return moved_distance


def calculate_zielfunktion(centroids, assignment):
    labels = range(len(centroids))
    data = list(assignment.keys())
    zielfunktionswert = 0
    for label in labels:
        assigned_points = [point for point in data if assignment.get(point) == label]
        if len(assigned_points) > 0:
            centroid = centroids[label]
            distances = np.array(all_dist(assigned_points, centroid))
            zielfunktionswert += (distances ** 2).sum()
        else:
            continue

    zielfunktionswert = round(zielfunktionswert, 4)
    return zielfunktionswert
