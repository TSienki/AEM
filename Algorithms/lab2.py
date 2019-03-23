import numpy as np


def random_groups(data_length, nclusters=20):
    """
    :param data_length:
    :param nclusters: number of clusters
    :return: list of pairs cluster number and data number
    """
    data_permutation_indices = np.random.permutation(np.linspace(0, data_length - 1, data_length, dtype=np.int))
    i = 0
    clusters = []
    for index in data_permutation_indices:
        if i == nclusters:
            i = 0
        clusters.append([i, index])
        i += 1
    return np.array(clusters)


def greedy_algorithm(clusters, dist_matrix, neighbourhood):
    # TODO: It doesn't work properly
    number_clusters = np.max(clusters[:, 0])
    for i in range(number_clusters + 1):
        first = True
        cluster_indices = clusters[clusters[:, 0] == i][:, 1]
        is_other_cluster = np.ones(dist_matrix.shape[0])
        is_other_cluster[cluster_indices] = 0
        costs = np.sum(count_costs(clusters, dist_matrix, number_clusters))

        for point_index in cluster_indices:
            if not first:
                dist_from_point = dist_matrix[point_index]
                neighbourhood_indices = np.argwhere((is_other_cluster == 1) &
                                                    (dist_from_point < neighbourhood)).reshape(-1)
                for neighbourhood_index in neighbourhood_indices:
                    if count_delta_cost(i, point_index, neighbourhood_index, clusters, dist_matrix):
                        print(f"Change {point_index} to {point_index}")
            else:
                first = False
    print(costs)


def change_cluster(first_index, second_index, dist_matrix):
    first_row = dist_matrix[first_index]
    second_row = dist_matrix[second_index]

    dist_matrix[first_index] = second_row
    dist_matrix[second_index] = first_row


def steepest_algorithm(cluster_indices, dist_matrix):
    # TODO: Implement it
    raise NotImplementedError


def find_neighbourhood_indices():
    # TODO: Implement it
    raise NotImplementedError


def count_delta_cost(first_point_cluster, first_point, second_point, clusters, dist_matrix):
    first_cluster_indices = clusters[clusters[:, 0] == first_point_cluster][:, 1]
    second_point_cluster = clusters[clusters[:, 1] == second_point][0, 0]
    second_cluster_indices = clusters[clusters[:, 0] == second_point_cluster][:, 1]
    cost_before_change = (count_cost_for_group(first_cluster_indices, dist_matrix) +
                          count_cost_for_group(second_cluster_indices, dist_matrix))
    first_cluster_indices[first_cluster_indices == first_point] = second_point
    second_cluster_indices[second_cluster_indices == second_point] = first_point
    cost_after_change = (count_cost_for_group(first_cluster_indices, dist_matrix) +
                         count_cost_for_group(second_cluster_indices, dist_matrix))
    return cost_after_change - cost_before_change


def count_costs(clusters, dist_matrix, number_clusters=None):
    costs = []
    if number_clusters == None:
        number_clusters = np.max(clusters[:, 0])
    for i in range(number_clusters + 1):
        cluster = clusters[clusters[:, 0] == i][:, 1]
        costs.append(count_cost_for_group(cluster, dist_matrix))
    return costs


def count_cost_for_group(cluster_points, dist_matrix):
    cost = 0
    for point in cluster_points:
        cost += count_cost_for_one_point(point, cluster_points, dist_matrix)
    return cost


def count_cost_for_one_point(point, group, dist_matrix):
    return np.sum(dist_matrix[point, group])
