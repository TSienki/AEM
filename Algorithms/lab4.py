import random
import numpy as np
from Lab1 import create_n_trees_kruskal, create_n_trees_prim
from Utilities.DataPreprocess import parse_data, create_dist_function, create_clusters_from_tree
import operator
import time

def cost_function(distance_matrix, groups):
    groups = [np.argwhere(groups == group_number).reshape(-1) for group_number in range(np.max(groups) + 1)]
    sum_ = np.sum([np.sum(distance_matrix[group, :][:, group]) / 2
                   for group in groups])
    pairs = np.sum([len(group)*(len(group)-1)/2 for group in groups])
    return sum_ / pairs, sum_, pairs


def new_cost(clusters, distance_matrix, prev_cost, point, point_group, neighbour_group, cached_gain=None):
    group = np.argwhere(clusters == point_group)
    new_group = np.argwhere(clusters == neighbour_group)

    old_pairs = len(group) - 1
    new_pairs = len(new_group)

    if cached_gain is not None:
        sum_ = prev_cost[1] - cached_gain
    else:
        distance_loss = np.sum(distance_matrix[group, point])
        distance_gain = np.sum(distance_matrix[new_group, point])
        sum_ = prev_cost[1] + distance_gain - distance_loss
    pairs = prev_cost[2] + new_pairs - old_pairs
    return sum_ / pairs, sum_, pairs


def count_group_gain(clusters, distance_matrix, point, point_group, neighbour_group):
    group = np.argwhere(clusters == point_group)
    new_group = np.argwhere(clusters == neighbour_group)
    distance_loss = np.sum(distance_matrix[group, point])
    distance_gain = np.sum(distance_matrix[new_group, point])
    return distance_loss - distance_gain


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


def get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point, candidates):
    cluster_indices = np.argwhere(clusters == clusters[point])
    is_other_cluster = np.ones(dist_matrix.shape[0])
    is_other_cluster[cluster_indices] = 0
    dist_from_point = dist_matrix[point].reshape(-1)
    neighbourhood_indices = np.argwhere(
        (is_other_cluster == 1) & (dist_from_point < neighbourhood_radius)).reshape(-1)
    if candidates:
        neighbourhood_indices = find_candidates(clusters, neighbourhood_indices)
    random.shuffle(neighbourhood_indices)
    return neighbourhood_indices


def run_algorithm_steepest(clusters, dist_matrix, neighbourhood_radius, candidates=False, cache=False):
    cost = cost_function(dist_matrix, clusters)

    for i in range(50):
        changes = 0
        for point in range(dist_matrix.shape[0]):
            cachedict = {}
            neighbourhood_indices = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius,
                                                      point, candidates)
            # print("Steepest,", len(neighbourhood_indices))
            best_neighbour = None
            best_cost = cost
            for neighbour in neighbourhood_indices:
                cost_after_change = None
                if cache:
                    if (point, clusters[point], clusters[neighbour]) in cachedict:
                        if cachedict[(point, clusters[point], clusters[neighbour])] < 0:
                            continue
                    else:
                        cachedict[(point, clusters[point], clusters[neighbour])] = \
                            count_group_gain(clusters, dist_matrix, point, clusters[point], clusters[neighbour])
                    cost_after_change = new_cost(clusters, dist_matrix, cost, point, clusters[point],
                                                 clusters[neighbour],
                                                 cachedict[(point, clusters[point], clusters[neighbour])])
                else:
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


def find_candidates(clusters, potential_candidates):
    candidates = {}
    candidates_sum = {}
    for candidate in potential_candidates:
        if clusters[candidate] not in candidates:
            candidates[clusters[candidate]] = candidate
            candidates_sum[clusters[candidate]] = 1
        else:
            candidates_sum[clusters[candidate]] += 1
    sorted_candidates = sorted(candidates_sum.items(), key=operator.itemgetter(1))
    amount = len(sorted_candidates) // 2
    groups = [x[0] for x in sorted_candidates[-amount:]]
    return [candidates[x] for x in groups]


def small_perturbations(clusters, number_of_perturbations, neighbourhood_radius, dist_matrix):
    # in each perturbation a random point's group is changed to another group that none of the neighbours belongs to
    for i in range(number_of_perturbations):
        point_to_perturbate = np.random.choice(len(clusters), 1)
        point_neighbourhood = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point_to_perturbate, True)
        target_group = np.random.choice([i for i in range(np.max(clusters))
                                         if i not in [clusters[x] for x in point_neighbourhood]
                                        and i != clusters[point_to_perturbate]], 1)
        clusters[point_to_perturbate] = target_group
    return clusters


def big_random_perturbation(clusters, percent_to_remove, dist_matrix):
    # Destroy - remove 10-30% of points
    amount_to_remove = int(len(clusters) * percent_to_remove/100)
    points_to_remove = np.random.choice([i for i in range(len(clusters))], amount_to_remove)
    for p in points_to_remove:
        clusters[p] = -1

    # Repair - add a point to group so that the average distance from this point to other points in the group is minimum
    for removed_point in points_to_remove:
        best_avg = np.inf
        best_group = -1
        for i in range(np.max(clusters)):
            distance = 0
            group_points = np.argwhere(clusters == i)
            for p in group_points:
                distance += dist_matrix[p, removed_point]
            avg_distance = distance/len(group_points)

            if avg_distance < best_avg:
                best_avg = avg_distance
                best_group = i
        clusters[removed_point] = best_group
    return clusters


def msls(dist_matrix, neighbourhood_radius, data, candidates=False, cache=False, option="random"):
    np.random.seed(0)
    best_clusters = None
    best_cost = np.inf
    cluster_before_best = None

    for i in range(1):
        clusters = np.ones(len(data), dtype=np.int32) * (-1)

        if option == "prim":
            result = create_n_trees_prim(data, dist_matrix, 20, 1)
            clusters = create_clusters_from_tree(data, result)
        elif option == "kruskal":
            result = create_n_trees_kruskal(data, dist_matrix, 20, 1)
            clusters = create_clusters_from_tree(data, result)
        elif option == "random":
            clusters = random_groups(data.shape[0])

        clusters_before = np.copy(clusters)

        run_algorithm_steepest(clusters, dist_matrix, neighbourhood_radius, candidates, cache)
        cost = cost_function(dist_matrix, clusters)[0]

        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters
            cluster_before_best = clusters_before
    return best_cost, best_clusters, cluster_before_best


def ils(dist_matrix, neighbourhood_radius, data, time_limit, candidates=False, cache=False,
        option="random", perturbation = "small"):
    timeout = time_limit
    timeout_start = time.time()
    np.random.seed(0)
    best_clusters = None
    best_cost = np.inf
    cluster_before_best = None
    clusters = np.ones(len(data), dtype=np.int32) * (-1)
    if option == "prim":
        result = create_n_trees_prim(data, dist_matrix, 20, 1)
        clusters = create_clusters_from_tree(data, result)
    elif option == "kruskal":
        result = create_n_trees_kruskal(data, dist_matrix, 20, 1)
        clusters = create_clusters_from_tree(data, result)
    elif option == "random":
        clusters = random_groups(data.shape[0])
    clusters_before = np.copy(clusters)

    while time.time() < timeout_start + timeout:
        run_algorithm_steepest(clusters, dist_matrix, neighbourhood_radius, candidates, cache)
        cost = cost_function(dist_matrix, clusters)[0]
        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters
            cluster_before_best = clusters_before
        print(cost)
        if perturbation == "small":
            clusters = small_perturbations(clusters, 20, 50, dist_matrix)
        elif perturbation == "big":
            clusters = big_random_perturbation(clusters, 30, dist_matrix)
    return best_cost, best_clusters, cluster_before_best
