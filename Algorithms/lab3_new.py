import numpy as np
from Algorithms.lab2_new import cost_function, get_neighbourhood, new_cost


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


def run_algorithm_steepest(clusters, dist_matrix, neighbourhood_radius, additional_method="none"):

    """
     :param clusters:
     :param dist_matrix:
     :param neighbourhood:
     :param additional_method:
     """
    cost = cost_function(dist_matrix, clusters)
    old_costs_cashed = None
    if additional_method in {"cache", "candidates_with_cache"}:
        old_costs_cashed = np.full(clusters.shape, -1)
    for _ in range(50):  # it prevents situation if steepest algorithm will be stuck
        changes = 0
        for i in range(np.max(clusters) + 1):
            is_first = True
            cluster_indices = np.argwhere(clusters == i)

            for point in cluster_indices:
                cluster_indices = np.argwhere(clusters == i)
                is_other_cluster = np.ones(dist_matrix.shape[0])
                is_other_cluster[cluster_indices] = 0

                if is_first:
                    is_first = False
                else:
                    dist_from_point = dist_matrix[point].reshape(-1)
                    potential_neighbourhood_indices = np.argwhere((is_other_cluster == 1)
                                                                  & (dist_from_point < neighbourhood_radius))
                    if additional_method in {"candidates", "candidates_with_cache"}:
                        reference_point = np.random.choice([list(point_indices)[0] for point_indices in cluster_indices
                                                            if point_indices not in potential_neighbourhood_indices])
                        candidates = find_candidates(reference_point, point, potential_neighbourhood_indices,
                                                     dist_matrix)

                    if additional_method == "cache":
                        change = steepest_algorithm_cache(clusters, point, dist_matrix,
                                                          neighbourhood_radius, old_costs_cashed, additional_method)
                    elif additional_method == "candidates":
                        change = algorithm_steepest(clusters, point, dist_matrix, candidates, additional_method)
                    elif additional_method == "candidates_with_cache":
                        change = steepest_algorithm_cache(clusters, point, dist_matrix,
                                                          candidates, old_costs_cashed, additional_method)
                    elif additional_method == "none":
                        change = algorithm_steepest(clusters, point, dist_matrix, neighbourhood_radius,
                                                    additional_method)

            if additional_method == "candidates":
                neighbourhood_indices = neighbourhood_radius
            else:
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


def algorithm_steepest(clusters, point, dist_matrix, neighbourhood_radius, additional_method="none"):
    cost = cost_function(dist_matrix, clusters)
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
        return True
    return False


def steepest_algorithm_cache(clusters, point, dist_matrix, neighbourhood_radius, old_costs_cashed, additional_method="none"):
    cost = cost_function(dist_matrix, clusters)

    if additional_method == "cache":
            smallest_after = np.inf
            best_neighbourhood_cost = None
            best_first_point_cost = None
            best_first_cluster_indices = None
            best_second_cluster_indices = None
            best_neighbour = None
            neighbourhood_indices = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point)
            # print("Steepest,", len(neighbourhood_indices))
            best_neighbour = None
            best_cost = cost

            for neighbourhood_index in neighbourhood_indices:
                first_cluster = clusters[point]
                first_cluster_indices = np.argwhere(clusters == first_cluster)
                second_cluster = clusters[neighbourhood_index]
                second_cluster_indices = np.argwhere(clusters == second_cluster)

                after_first_cluster_indices = first_cluster_indices[first_cluster_indices != point]
                after_second_cluster_indices = second_cluster_indices[second_cluster_indices != neighbourhood_index]
                if old_costs_cashed[point] == -1:
                    first_point_cost_before = count_cost_for_one_point(point, first_cluster_indices,
                                                                       dist_matrix)
                    old_costs_cashed[point] = first_point_cost_before
                else:
                    first_point_cost_before = old_costs_cashed[point]

                if old_costs_cashed[neighbourhood_index] == -1:
                    neighbourhood_cost_before = count_cost_for_one_point(neighbourhood_index,
                                                                         second_cluster_indices, dist_matrix)
                    old_costs_cashed[neighbourhood_index] = neighbourhood_cost_before
                else:
                    neighbourhood_cost_before = old_costs_cashed[neighbourhood_index]

                cost_before_change = first_point_cost_before + neighbourhood_cost_before

                neighbourhood_cost_after = count_cost_for_one_point(neighbourhood_index,
                                                                    after_first_cluster_indices, dist_matrix)
                first_point_cost_after = count_cost_for_one_point(point, after_second_cluster_indices,
                                                                  dist_matrix)
                cost_after_change = neighbourhood_cost_after + first_point_cost_after
                # print(cost_after_change, cost_before_change)
                if cost_before_change > cost_after_change and smallest_after > cost_after_change:
                    best_neighbour = neighbourhood_index[0]
                    best_cost = cost_after_change
                    best_neighbourhood_cost = neighbourhood_cost_after
                    best_first_point_cost = first_point_cost_after
                    best_first_cluster_indices = np.copy(after_first_cluster_indices)
                    best_second_cluster_indices = np.copy(after_second_cluster_indices)
                    # print(best_neighbourhood_cost, best_first_point_cost)

            if best_neighbour is not None:
                old_costs_cashed[best_first_cluster_indices] = -1
                old_costs_cashed[best_second_cluster_indices] = -1
                old_costs_cashed[point] = best_first_point_cost
                old_costs_cashed[best_neighbour] = best_neighbourhood_cost

                clusters[point] = clusters[best_neighbour]
                cost = best_cost
                return True
            return False
    else:
            for point in range(dist_matrix.shape[0]):
                neighbourhood_indices = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point)
                # print("Steepest,", len(neighbourhood_indices))
                best_neighbour = None
                best_cost = cost

                for neighbour in neighbourhood_indices:
                    cost_after_change = new_cost(clusters, dist_matrix, cost, point, clusters[point],
                                                 clusters[neighbour])
                    if best_cost[0] > cost_after_change[0]:
                        if np.count_nonzero(clusters == clusters[point]) > 1:
                            best_cost = cost_after_change
                            best_neighbour = neighbour

                if best_neighbour is not None:
                    clusters[point] = clusters[best_neighbour]
                    cost = best_cost


def find_candidates(reference_point, current_point, potential_candidates, dist_matrix):
    boundary_distance = dist_matrix[reference_point, current_point]
    candidates = []
    for i, candidate in enumerate(potential_candidates):
        candidate_distance = dist_matrix[reference_point, candidate]
        if candidate_distance < boundary_distance:
            candidates.append((candidate[0], candidate_distance[0]))

    # Sorting by increasing distance from reference point
    for i in range(0, len(candidates)):
        for j in range(0, len(candidates) - i - 1):
            if candidates[j][1] > candidates[j + 1][1]:
                temp = candidates[j]
                candidates[j] = candidates[j + 1]
                candidates[j + 1] = temp
    return [[x[0]] for x in candidates[0:7]]


def count_cost_for_one_point(point, group, dist_matrix):
    return np.sum(dist_matrix[point, group])
