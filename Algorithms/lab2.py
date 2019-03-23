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


def greedy_algorithm(clusters, dist_matrix, neighbourhood):
    number_clusters = np.max(clusters)
    cost_before = np.sum(count_costs(clusters, dist_matrix, number_clusters))
    for j in range(100):
        # TODO implement stop criterion instead of this loop
        for i in range(number_clusters + 1):
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
                    neighbourhood_indices = np.argwhere((is_other_cluster == 1) &
                                                        (dist_from_point < neighbourhood))

                    for neighbourhood_index in neighbourhood_indices:
                        before, after = count_delta_cost(point_index, neighbourhood_index, clusters, dist_matrix)
                        if before > after:
                            temp = clusters[point_index]
                            clusters[point_index] = clusters[neighbourhood_index]
                            clusters[neighbourhood_index] = temp
                            break
    cost_after = np.sum(count_costs(clusters, dist_matrix, number_clusters))
    print(cost_before - cost_before, cost_after, cost_before)


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
    return cost


def count_cost_for_one_point(point, group, dist_matrix):
    return np.sum(dist_matrix[point, group])
