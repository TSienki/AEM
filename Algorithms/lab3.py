import numpy as np


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


def run_algorithm(clusters, dist_matrix, neighbourhood, additional_method="none"):
    """
    :param clusters:
    :param dist_matrix:
    :param neighbourhood:
    :param additional_method:
    """
    old_costs_cashed = None
    if additional_method == "cache":
        old_costs_cashed = np.full(clusters.shape, -1)
    for _ in range(10): # it prevents situation if steepest algorithm will be stuck
        changes = 0
        for i in range(np.max(clusters) + 1):
            is_first = True
            cluster_indices = np.argwhere(clusters == i)

            for point_index in cluster_indices:
                cluster_indices = np.argwhere(clusters == i)
                is_other_cluster = np.ones(dist_matrix.shape[0])
                is_other_cluster[cluster_indices] = 0

                if is_first:
                    is_first = False
                else:
                    dist_from_point = dist_matrix[point_index].reshape(-1)
                    neighbourhood_indices = np.argwhere((is_other_cluster == 1) & (dist_from_point < neighbourhood))

                    change = steepest_algorithm(neighbourhood_indices,
                                              point_index,
                                              clusters,
                                              dist_matrix,
                                              old_costs_cashed,
                                              additional_method)
                    if change:
                        changes += 1

        print(changes)
        if changes == 0:
            break


def change_cluster(first_index, second_index, dist_matrix):
    first_row = dist_matrix[first_index]
    second_row = dist_matrix[second_index]

    dist_matrix[first_index] = second_row
    dist_matrix[second_index] = first_row


def greedy_algorithm(neighbourhood_indices, point_index, clusters, dist_matrix, old_costs_cashed, additional_method):
    for neighbourhood_index in neighbourhood_indices:
        if additional_method == "cache":
            first_cluster = clusters[point_index]
            first_cluster_indices = np.argwhere(clusters == first_cluster)
            second_cluster = clusters[neighbourhood_index]
            second_cluster_indices = np.argwhere(clusters == second_cluster)

            after_first_cluster_indices = first_cluster_indices[first_cluster_indices != first_cluster]
            after_second_cluster_indices = second_cluster_indices[second_cluster_indices != second_cluster]
            if old_costs_cashed[point_index] == -1:
                first_point_cost_before = count_cost_for_one_point(point_index, first_cluster_indices, dist_matrix)
                old_costs_cashed[point_index] = first_point_cost_before
            else:
                first_point_cost_before = old_costs_cashed[point_index]

            if old_costs_cashed[neighbourhood_index] == -1:
                neighbourhood_cost_before = count_cost_for_one_point(neighbourhood_index, second_cluster_indices, dist_matrix)
                old_costs_cashed[neighbourhood_index] = neighbourhood_cost_before
            else:
                neighbourhood_cost_before = old_costs_cashed[neighbourhood_index]

            cost_before_change = (first_point_cost_before + neighbourhood_cost_before)

            neighbourhood_cost_after = count_cost_for_one_point(neighbourhood_index, after_first_cluster_indices, dist_matrix)
            first_point_cost_after = count_cost_for_one_point(point_index, after_second_cluster_indices, dist_matrix)
            cost_after_change = neighbourhood_cost_after + first_point_cost_after

            if cost_before_change > cost_after_change:
                old_costs_cashed[point_index] = first_point_cost_after
                old_costs_cashed[neighbourhood_index] = neighbourhood_cost_after
                change_point_in_clusters(clusters, point_index, neighbourhood_index)
                return True
        else:
            before, after = count_delta_cost(point_index, neighbourhood_index, clusters, dist_matrix)
            if before > after:
                change_point_in_clusters(clusters, point_index, neighbourhood_index)
                return True
    return False


def steepest_algorithm(neighbourhood_indices, point_index, clusters, dist_matrix, old_costs_cashed, additional_method):
    smallest_after = np.inf
    smallest_neighbour = -1


    if additional_method == "cache":
        neighbourhood_cost_after = None
        first_point_cost_after = None
        neighbourhood_index = None
        first_cluster_indices = None
        second_cluster_indices = None

        for neighbourhood_index in neighbourhood_indices:
            first_cluster = clusters[point_index]
            first_cluster_indices = np.argwhere(clusters == first_cluster)
            second_cluster = clusters[neighbourhood_index]
            second_cluster_indices = np.argwhere(clusters == second_cluster)

            after_first_cluster_indices = first_cluster_indices[first_cluster_indices != first_cluster]
            after_second_cluster_indices = second_cluster_indices[second_cluster_indices != second_cluster]
            if old_costs_cashed[point_index] == -1:
                first_point_cost_before = count_cost_for_one_point(point_index, first_cluster_indices, dist_matrix)
                old_costs_cashed[point_index] = first_point_cost_before
            else:
                first_point_cost_before = old_costs_cashed[point_index]

            if old_costs_cashed[neighbourhood_index] == -1:
                neighbourhood_cost_before = count_cost_for_one_point(neighbourhood_index, second_cluster_indices, dist_matrix)
                old_costs_cashed[neighbourhood_index] = neighbourhood_cost_before
            else:
                neighbourhood_cost_before = old_costs_cashed[neighbourhood_index]

            cost_before_change = first_point_cost_before + neighbourhood_cost_before

            neighbourhood_cost_after = count_cost_for_one_point(neighbourhood_index, after_first_cluster_indices, dist_matrix)
            first_point_cost_after = count_cost_for_one_point(point_index, after_second_cluster_indices, dist_matrix)
            cost_after_change = neighbourhood_cost_after + first_point_cost_after
            # print(cost_after_change, cost_before_change)
            if cost_before_change > cost_after_change and smallest_after > cost_after_change:
                smallest_neighbour = neighbourhood_index
                smallest_after = cost_after_change
        if smallest_neighbour > -1:
            old_costs_cashed[first_cluster_indices] = -1
            old_costs_cashed[second_cluster_indices] = -1
            old_costs_cashed[point_index] = first_point_cost_after
            old_costs_cashed[neighbourhood_index] = neighbourhood_cost_after
            change_point_in_clusters(clusters, point_index, smallest_neighbour)
            return True
    else:
        for neighbourhood_index in neighbourhood_indices:
            before, after = count_delta_cost(point_index, neighbourhood_index, clusters, dist_matrix)
            if before > after and smallest_after > after:
                smallest_neighbour = neighbourhood_index
                smallest_after = after
        if smallest_neighbour > -1:
            change_point_in_clusters(clusters, point_index, smallest_neighbour)
            return True
    return False


def change_point_in_clusters(clusters, point_1, point_2):
    temp = clusters[point_1]
    clusters[point_1] = clusters[point_2]
    clusters[point_2] = temp


def count_delta_cost(first_point, second_point, clusters, dist_matrix):
    #It is defined as difference between cost before change and potential cost after change.
    first_cluster = clusters[first_point]
    second_cluster = clusters[second_point]
    first_cluster_indices = np.argwhere(clusters == first_cluster)
    second_cluster_indices = np.argwhere(clusters == second_cluster)
    cost_before_change = (count_cost_for_one_point(first_point, first_cluster_indices, dist_matrix) +
                          count_cost_for_one_point(second_point, second_cluster_indices, dist_matrix))

    after_first_cluster_indices = first_cluster_indices[first_cluster_indices != first_cluster]
    after_second_cluster_indices = second_cluster_indices[second_cluster_indices != second_cluster]
    cost_after_change = (count_cost_for_one_point(second_point, after_first_cluster_indices, dist_matrix) +
                          count_cost_for_one_point(first_point, after_second_cluster_indices, dist_matrix))
    return cost_before_change, cost_after_change


def count_costs(clusters, dist_matrix, number_clusters=None):
    costs = []
    if number_clusters == None:
        number_clusters = np.max(clusters[:, 0])
    for i in range(number_clusters + 1):
        cluster = np.argwhere(clusters == i)
        costs.append(count_cost_for_group(cluster, dist_matrix))
    return costs


def count_cost_for_group(cluster_points, dist_matrix):
    cost = 0
    for point in cluster_points:
        cost += count_cost_for_one_point(point, cluster_points, dist_matrix)
    if len(cluster_points) == 0:
        cost = 0
    else:
        cost = cost/(cluster_points.shape[0] * (cluster_points.shape[0] - 1))
    return cost


def count_cost_for_one_point(point, group, dist_matrix):
    return np.sum(dist_matrix[point, group])
