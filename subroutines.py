import numpy as np
import pandas as pd
import random

from helpers import *


def initialise_centroids(data, k):
    """
    Creates k points uniformly distributed in the [0, 120]^d cube, where d is the dimension of the points in data. Uses
    a fixed seed, to produce reproducible results.

    :param data: Should be a non-empty list of tuples. Is used to determine the dimension to use.
    :param k: Number of points to be created
    :return: A list containing k tuples of the same length as the first tuple in data.
    """
    dimension = len(data[0])
    np.random.seed(0)
    centroids = np.random.rand(k, dimension)
    centroids *= 120  # die Daten liegen nicht ganz in einem 100^n Würfel
    centroids = [tuple(row) for row in centroids]
    return centroids


def update_centroids(centroids, assignment):
    """
    For each centroid, takes all assigned points and computes their center of mass. This point is the new centroid and
    replaces the old one. This is done internally without returning the new centroids. In addition, the distance between
    the old and the new center is computed and returned.

    :param centroids: The old centroids (a list of tuples of same dimension). This function updates them internally,
        without returning them.
    :param assignment: The usual assignment dict. A dictionary which contains the current label for each data point.
    :return: A pd.Series containing the distances the centroids have been moved by this function call. It has the
        default index, which corresponds to the labels the centroids represent.
    """
    labels = range(len(centroids))
    data = list(assignment.keys())
    moved_distance = list()
    empty_group_count = 0
    for label in labels:
        assigned_points = [point for point in data if assignment.get(point) == label]
        if len(assigned_points) == 0:
            # new_centroid = data[random.randint(0, len(data) - 1)]
            new_centroid = data[empty_group_count]
            moved_distance.append(dist(centroids[label], new_centroid))
            centroids[label] = new_centroid
            empty_group_count += 1
            continue
        arr = np.array(assigned_points)
        new_centroid = arr.sum(axis=0) / len(assigned_points)
        moved_distance.append(dist(centroids[label], new_centroid))
        centroids[label] = tuple(new_centroid)

    moved_distance = pd.Series(moved_distance)
    return moved_distance


def update_centers_to_pivot_distances(centroids, pivot, saved_dist_comp_theory=None, saved_dist_comp_practice=None):
    """
    Computes the distances from the centroids to the pivot object. If provided decreases the number of saved distance
    computations by the number of distances computed by this function call, which is the length of centroids.

    :param centroids: A list of points (tuples of floats) of the same dimension.
    :param pivot: A single point (tuple of floats) of the same dimension as centroids.
    :param saved_dist_comp_theory: If provided it will be updated internally.
    :param saved_dist_comp_practice: If provided it will be updated internally.
    :return: A dictionary. The keys are the labels of the centers and the values are the distances to the query.
    """
    distances_temp = all_dist(dataset=centroids, query=pivot)
    if saved_dist_comp_theory is not None:
        saved_dist_comp_theory -= len(centroids)
    if saved_dist_comp_practice is not None:
        saved_dist_comp_practice -= len(centroids)

    return dict(zip(distances_temp.index, distances_temp))


def update_points_to_pivot_distances(points, pivot, saved_dist_comp_theory=None, saved_dist_comp_practice=None):
    """
    Computes the distances from the points to the pivot object. If provided, decreases the number of saved distance
    computations by the number of distances computed by this function call, which is the length of points.

    :param points: A list of points (tuples of floats) of the same dimension.
    :param pivot: A single point (tuple of floats) of the same dimension as points.
    :param saved_dist_comp_theory: If provided it will be updated internally.
    :param saved_dist_comp_practice: If provided it will be updated internally.
    :return: A dictionary. The keys are the points themselves and the values are the distances to the query.
    """
    distances_temp = all_dist(dataset=points, query=pivot)
    if saved_dist_comp_theory is not None:
        saved_dist_comp_theory -= len(points)
    if saved_dist_comp_practice is not None:
        saved_dist_comp_practice -= len(points)

    return dict(zip(points, distances_temp))


def calculate_zielfunktion(centroids, assignment):
    """
    Evaluates the objective function of the k-Means problem for the given centers and assignment. Which is the squared
    sum of the points and their assigned centers.

    :param centroids: A list of points (tuples of floats) of the same dimension.
    :param assignment: The usual assignment dict. A dictionary which contains the current label for each data point.
    :return: The value of the objective function for the given inputs rounded to its 4th decimal.
    """
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
