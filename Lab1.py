import numpy as np
import matplotlib.pyplot as plt
import MST
from prim import prim, prim_n_tree_generate
import random


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


def create_one_tree(data, dist_matrix):
    starting_point = random.randint(0, len(data))
    result = prim(dist_matrix, starting_point)

    visualise_tree(data, result, starting_point)


def create_n_trees(data, dist_matrix):
    starting_points_indices = np.random.choice(data.shape[0],
                                               10,
                                               replace=False)
    starting_points_indices = np.sort(starting_points_indices)
    mask = np.ones(data.shape[0], dtype=bool)
    mask[starting_points_indices] = False

    result = prim_n_tree_generate(dist_matrix, starting_points_indices)
    visualise_n_trees(data, result, mask)


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
    create_one_tree(data, dist_matrix)
    # create_n_trees(data, dist_matrix)


if __name__ == "__main__":
    main_function()
