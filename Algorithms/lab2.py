import numpy as np


def random_groups(data_length, clusters=20):
    data_permutation_indices = np.random.permutation(np.linspace(0, data_length - 1, data_length, dtype=np.int))
    grouped_data_permutation_indices = []
    i = 0
    first_iteration = True
    for index in data_permutation_indices:
        if i == clusters:
            i = 0
            first_iteration = False
        if first_iteration:
            grouped_data_permutation_indices.append([])
        grouped_data_permutation_indices[i].append(index)
        i += 1
    return grouped_data_permutation_indices


def greedy_algorithm(clusters_indices, dist_matrix, neighbourhood):
    # TODO: It doesn't work properly
    costs = count_costs(clusters_indices, dist_matrix)
    for cluster_indices in clusters_indices:
        first = True
        np_cluster_indices = np.array(cluster_indices)
        others_clusters = np.ones(dist_matrix.shape[0])
        others_clusters[np_cluster_indices] = 0
        for index in np_cluster_indices:
            print(index)
            cost_before = count_cost_for_one_point(index, np_cluster_indices, dist_matrix)
            cluster_without_current_point = np_cluster_indices[np_cluster_indices != index]
            if not first:
                dist_from_point = dist_matrix[index]
                neighbourhood_indices = np.argwhere((others_clusters == 1) & (dist_from_point < neighbourhood))

                for neighbourhood_index in neighbourhood_indices:
                    new_cost = count_cost_for_one_point(neighbourhood_index,
                                                        cluster_without_current_point,
                                                        dist_matrix)
                    if new_cost < cost_before:
                        change_cluster(index, neighbourhood_index, clusters_indices)
                        break
            else:
                first = False
    print(costs)


def change_cluster(first_index, second_index, clusters):
    for cluster in clusters:
        for i, index in enumerate(cluster):
            if first_index == index:
                cluster.pop(i)
                cluster.append(i)
                break
            if second_index == index:
                cluster.pop(i)
                cluster.append(i)
                break


def steepest_algorithm(cluster_indices, dist_matrix):
    # TODO: Implement it
    raise NotImplementedError


def find_neighbourhood_indices():
    # TODO: Implement it
    raise NotImplementedError


def count_costs(clusters_indices, dist_matrix):
    costs = []
    for cluster in clusters_indices:
        cluster_np = np.array(cluster)
        costs.append(count_cost_for_group(cluster_np, dist_matrix))
    return costs


def count_cost_for_group(cluster_points, dist_matrix):
    cost = 0
    for point in cluster_points:
        cost += count_cost_for_one_point(point, cluster_points, dist_matrix)
    return cost


def count_cost_for_one_point(point, group, dist_matrix):
    return np.sum(dist_matrix[point, group])
