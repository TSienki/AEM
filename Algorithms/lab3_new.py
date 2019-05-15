import random
import numpy as np
import operator


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
        reference_point = np.random.choice([list(point_indices)[0] for point_indices in cluster_indices
                                            if point_indices not in neighbourhood_indices])
        # neighbourhood_indices = find_candidates(reference_point, point, neighbourhood_indices, dist_matrix)
        neighbourhood_indices = find_candidates(clusters, reference_point, point, neighbourhood_indices, dist_matrix)
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


def find_candidates(clusters, reference_point, current_point, potential_candidates, dist_matrix):
    # boundary_distance = dist_matrix[reference_point, current_point]
    # candidates = []
    # for i, candidate in enumerate(potential_candidates):
    #     candidate_distance = dist_matrix[reference_point, candidate]
    #     if candidate_distance < boundary_distance:
    #         candidates.append((candidate, candidate_distance))
    # # Sorting by increasing distance from reference point
    # for i in range(0, len(candidates)):
    #     for j in range(0, len(candidates) - i - 1):
    #         if candidates[j][1] > candidates[j + 1][1]:
    #             temp = candidates[j]
    #             candidates[j] = candidates[j + 1]
    #             candidates[j + 1] = temp
    # return [x[0] for x in candidates[0:7]]
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

    # if len(candidates_sum) is not 0:
    #     max_key = max(candidates_sum.items(), key=operator.itemgetter(1))[0]
    #     candidates_to_return.append(candidates[max_key])
    # for key, value in candidates_sum.items():
    #     if value > 2:
    #         candidates_to_return.append(candidates[key])
    return [candidates[x] for x in groups]
