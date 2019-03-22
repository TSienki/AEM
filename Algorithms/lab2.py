import numpy as np


def random_groups(data_length, clusters=20):
    data_permutation_indexes = np.random.permutation(np.linspace(0, data_length - 1, data_length, dtype=np.int))
    grouped_data_permutation_indexes = []
    i = 0
    first_iteration = True
    for index in data_permutation_indexes:
        if i == 20:
            i = 0
            first_iteration = False
        if first_iteration:
            grouped_data_permutation_indexes.append([])
        grouped_data_permutation_indexes[i].append(index)
        i += 1
    return grouped_data_permutation_indexes


def greedy_algorithm(points, clusters_indices):
    # TODO: Implement it
    raise NotImplementedError


def steepest_algorithm(points, cluster_indices):
    # TODO: Implement it
    raise NotImplementedError


def find_neighbourhoods():
    # TODO: Implement it
    raise NotImplementedError


def count_cost():
    # TODO: Implement it
    raise NotImplementedError
