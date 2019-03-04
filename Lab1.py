import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def create_matrix(data_, dist_fun):
    """
    :param data_: positions of points
    :param dist_fun: a function, which counts distance between two points
    :return: distance matrix
    """
    i, j = 0, 0
    matrix = np.zeros((data_.shape[0], data_.shape[0]))
    for point_1 in data_:
        j = 0
        for point_2 in data_:
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
    starting_points = 10
    # np.random.seed(19680801)

    starting_points_indices = np.random.choice(data.shape[0],
                                               starting_points,
                                               replace=False)
    mask = np.ones(data.shape[0], dtype=bool)
    mask[starting_points_indices] = False

    dist_matrix = create_matrix(data, euclidean_distance)

    plt.scatter(data[mask].T[0], data[mask].T[1])
    plt.scatter(data[~mask].T[0], data[~mask].T[1], c='r')
    plt.xlim(-10, 260)
    plt.ylim(-10, 260)
    plt.show()


if __name__ == "__main__":
    main_function()
