import random
from copy import deepcopy

import numpy as np


def cost_function(distance_matrix, groups):
    groups = [np.argwhere(groups == group_number).reshape(-1) for group_number in range(np.max(groups) + 1)]
    sum_ = np.sum([np.sum(distance_matrix[group, :][:, group]) / 2
                   for group in groups])
    pairs = np.sum([len(group)*(len(group)-1)/2 for group in groups])
    return sum_ / pairs, sum_, pairs


def new_cost(clusters, distance_matrix, prev_cost, point, point_group, neighbour_group):
    group = np.argwhere(clusters == point_group)
    new_group = np.argwhere(clusters == neighbour_group)

    distance_loss = np.sum(distance_matrix[group, point])
    distance_gain = np.sum(distance_matrix[new_group, point])
    old_pairs = len(group) - 1
    new_pairs = len(new_group)
    sum_ = prev_cost[1] + distance_gain - distance_loss
    pairs = prev_cost[2] + new_pairs - old_pairs
    return sum_ / pairs, sum_, pairs


def random_groups(data_length, nclusters=20):
    """
    :param data_length:
    :param nclusters: number of clusters
    :return: list of pairs cluster number and data number
    """
    data_permutation_indices = np.random.permutation(np.linspace(0, data_length - 1, data_length, dtype=np.int))
    i = 0
    clusters = np.ones(data_length, dtype=np.int32) * (-1)
    for index in data_permutation_indices:
        if i == nclusters:
            i = 0
        clusters[index] = i
        i += 1
    return clusters


def get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point):
    cluster_indices = np.argwhere(clusters == clusters[point])
    is_other_cluster = np.ones(dist_matrix.shape[0])
    is_other_cluster[cluster_indices] = 0

    dist_from_point = dist_matrix[point].reshape(-1)
    neighbourhood_indices = np.argwhere(
        (is_other_cluster == 1) & (dist_from_point < neighbourhood_radius)).reshape(-1)
    # print(neighbourhood_indices)
    # neighbourhood_indices = np.unique(neighbourhood_indices)
    # print(neighbourhood_indices)
    random.shuffle(neighbourhood_indices)
    return neighbourhood_indices


def run_algorithm_steepest(clusters, dist_matrix, neighbourhood_radius, additional_method="none"):
    cost = cost_function(dist_matrix, clusters)

    for i in range(50):
        changes = 0
        for point in range(dist_matrix.shape[0]):
            neighbourhood_indices = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point)
            # print("Steepest,", len(neighbourhood_indices))
            best_neighbour = None
            best_cost = cost

            for neighbour in neighbourhood_indices:
                cost_after_change = new_cost(clusters, dist_matrix, cost, point, clusters[point], clusters[neighbour])
                if best_cost[0] > cost_after_change[0]:
                    if np.count_nonzero(clusters == clusters[point]) > 1:
                        best_cost = cost_after_change
                        best_neighbour = neighbour

            if best_neighbour is not None:
                clusters[point] = clusters[best_neighbour]
                cost = best_cost
                changes += 1
        if changes == 0:
            break
            # print("break")


def run_algorithm_greedy(clusters, dist_matrix, neighbourhood_radius, additional_method="none"):
    cost = cost_function(dist_matrix, clusters)
    for i in range(50):
        changes = 0
        for point in range(dist_matrix.shape[0]):
            neighbourhood_indices = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point)
            # print("Greedy,", len(neighbourhood_indices))
            for neighbour in neighbourhood_indices:
                cost_after_change = new_cost(clusters, dist_matrix, cost, point, clusters[point], clusters[neighbour])
                if cost[0] > cost_after_change[0]:
                    clusters[point] = clusters[neighbour]
                    cost = cost_after_change
                    changes += 1
                    break
        if changes == 0:
            # [print(len(np.argwhere(clusters == group_number).reshape(-1))) for group_number in range(np.max(clusters) + 1)]
            break
