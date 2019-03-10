import numpy as np
import matplotlib.pyplot as plt
import kruskal
from prim import prim, prim_n_tree_generate
import random
import time


def euclidean_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def create_matrix(data, dist_fun):
    """
    :param data: positions of points
    :param dist_fun: a function, which counts distance between two points
    :return: distance matrix
    """
    i, j = 0, 0
    matrix = np.zeros((data.shape[0], data.shape[0]))
    for point_1 in data:
        j = 0
        for point_2 in data:
            distance = dist_fun(point_1, point_2)
            matrix[i][j] = distance
            j += 1
        i += 1
    return matrix


def time_measure(func, args_for_func):
    """
    :param func:
    :param args_for_func:
    :return: time in seconds
    """
    start = time.time()
    result = func(*args_for_func)
    end = time.time()
    return end - start, result


def time_measure_kruskal(func, g, dist_matrix):
    """
    :param func:
    :param args_for_func:
    :return: time in seconds
    """
    start = time.time()
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            g.add_edge(i, j, dist_matrix[i][j])
    result = func()
    end = time.time()
    return end - start, result


def create_n_trees_kruskal(data, dist_matrix, steps_for_time_measurements=1):
    times_measurements = []
    for i in range(steps_for_time_measurements):
        g = kruskal.Graph(np.size(dist_matrix))
        measurement, result = time_measure_kruskal(g.kruskal_mst, g, dist_matrix)
        print(measurement)
        del g
        times_measurements.append(measurement)

    cost = count_cost(result[:len(result)-9], dist_matrix)
    print(f"Minimalne drzewo rozpinajace: {cost}")

    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków to "
          f"min: {min(times_measurements)} sekund, max: {max(times_measurements)} sekund i "
          f"avg: {sum(times_measurements)/len(times_measurements)} sekund")
    for u, v, weight in result[:len(result)-9]:
         plt.plot([data[u].T[0], data[v].T[0]], [data[u].T[1], data[v].T[1]], 'k-')
    plt.scatter(data.T[0], data.T[1])
    plt.xlim(-10, 260)
    plt.ylim(-10, 260)
    plt.show()

def create_one_tree_prim(data, dist_matrix, steps_for_time_measurements=1):
    times_measurements = []
    for i in range(steps_for_time_measurements):
        starting_point = random.randint(0, len(data) - 1)
        measurement, result = time_measure(prim, (dist_matrix, starting_point))
        times_measurements.append(measurement)

    visualise_tree(data, result, starting_point)
    cost = count_cost(result, dist_matrix)
    print(f"Minimalne drzewo rozpinajace: {cost}")
    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków to "
          f"min: {min(times_measurements)} sekund, max: {max(times_measurements)} sekund i "
          f"avg: {sum(times_measurements)/len(times_measurements)} sekund")


def create_n_trees_prim(data, dist_matrix, steps_for_time_measurements=1):
    times_measurements = []
    for i in range(steps_for_time_measurements):
        starting_points_indices = np.random.choice(data.shape[0],
                                                   10,
                                                   replace=False)
        starting_points_indices = np.sort(starting_points_indices)
        mask = np.ones(data.shape[0], dtype=bool)
        mask[starting_points_indices] = False
        measurement, result = time_measure(prim_n_tree_generate, (dist_matrix, starting_points_indices))
        times_measurements.append(measurement)

    visualise_n_trees(data, result, mask)

    cost = count_cost(result, dist_matrix)
    print(f"Minimalne drzewo rozpinajace: {cost}")
    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków to "
          f"min: {min(times_measurements)} sekund, max: {max(times_measurements)} sekund i "
          f"avg: {sum(times_measurements)/len(times_measurements)} sekund")


def visualise_tree(points, nodes, starting_point):
    for node in nodes:
        plt.plot([points[node[0]][0], points[node[1]][0]],
                 [points[node[0]][1], points[node[1]][1]], 'k-')
    plt.scatter(points.T[0], points.T[1])
    plt.scatter(points[starting_point][0], points[starting_point][1], c='r')
    plt.xlim(-10, 260)
    plt.ylim(-10, 260)
    plt.show()


def visualise_n_trees(points, nodes, mask):
    for node in nodes:
        plt.plot([points[node[0]][0], points[node[1]][0]],
                 [points[node[0]][1], points[node[1]][1]], 'k-')
    plt.scatter(points[mask].T[0], points[mask].T[1])
    plt.scatter(points[~mask].T[0], points[~mask].T[1], c='r')
    plt.xlim(-10, 260)
    plt.ylim(-10, 260)
    plt.show()


def count_cost(nodes, cost_matrix):
    cost = 0
    for node in nodes:
        cost += cost_matrix[node[0], node[1]]
    return cost


def read_data(f_name):
    data = []
    with open(f_name) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            coordinates = line.split()
            data.append((int(coordinates[0]), int(coordinates[1])))
            i += 1

    return np.array(data)


def main_function():
    data = read_data("objects.data")
    dist_matrix = create_matrix(data, euclidean_distance)
    # create_one_tree_prim(data, dist_matrix, 100)
    create_n_trees_prim(data, dist_matrix, 100)
    # create_n_trees_kruskal(data, dist_matrix, 100)


if __name__ == "__main__":
    main_function()
