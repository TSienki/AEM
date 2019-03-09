import numpy as np
import matplotlib.pyplot as plt
import MST
from prim import prim
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
    starting_point = random.randint(0, len(data))
    dist_matrix = create_matrix(data, euclidean_distance)
    result = prim(dist_matrix, starting_point)

    for node in result:
        plt.plot([data[node[0]][0], data[node[1]][0]],
                 [data[node[0]][1], data[node[1]][1]], 'k-')

    plt.scatter(data.T[0], data.T[1])
    plt.scatter(data[starting_point][0], data[starting_point][1], c='r')
    plt.xlim(-10, 260)
    plt.ylim(-10, 260)
    plt.show()


if __name__ == "__main__":
    main_function()
