import itertools
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


def msls(dist_matrix, neighbourhood_radius, data, candidates=False, cache=False, option="random"):
    local_optimums = []
    local_optimums_costs = []
    current_step = 0

    while current_step < 5:
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

        if cost not in local_optimums_costs:
            local_optimums.append(clusters)
            local_optimums_costs.append(cost_function(dist_matrix, clusters)[0])
            current_step += 1
    return local_optimums, local_optimums_costs


def make_recombination(parents):
    child = np.ones_like(parents[0])*np.inf
    noclusters = np.max(parents[0])
    for pair in itertools.combinations(range(len(parents[0])), 2):
        if parents[0][pair[0]] == parents[0][pair[1]] and parents[1][pair[0]] == parents[1][pair[1]]:
            child[pair[0]] = parents[0][pair[0]]
            child[pair[1]] = parents[0][pair[1]]
    for id, value in enumerate(child):
        if value == np.inf:
            child[id] = np.random.randint(0, noclusters)
    return child.astype(int)


def evolutionary(dist_matrix, neighbourhood_radius, local_optimums, local_optimums_costs, time_limit):
    timeout = time_limit
    timeout_start = time.time()
    optimums = np.copy(local_optimums)
    optimums_costs = np.copy(local_optimums_costs)

    while time.time() < timeout_start + timeout:
        parents_ids = random.sample(range(0,len(optimums_costs)), 2)
        elite_optimum = optimums[parents_ids[0]]
        second_parent = optimums[parents_ids[1]]
        # elite_optimum = optimums[np.argmax(optimums_costs)]
        # second_parent_id = np.random.choice([x for x in range(len(optimums_costs)) if x != np.argmax(optimums_costs)])
        # second_parent = optimums[second_parent_id]

        child = make_recombination([elite_optimum, second_parent])
        # print(child)
        child_cost = cost_function(dist_matrix, child)[0]
        # print(f"Child cost before steepest: {child_cost}")

        run_algorithm_steepest(child, dist_matrix, neighbourhood_radius, candidates=False, cache=True)

        child_cost = cost_function(dist_matrix, child)[0]
        # print(f"Child cost after steepest: {child_cost}")
        worst_solution = np.argmax(optimums_costs)
        if child_cost < optimums_costs[worst_solution]:
            # print("One is smaller")
            optimums[worst_solution] = child
            optimums_costs[worst_solution] = child_cost
    return optimums, optimums_costs
