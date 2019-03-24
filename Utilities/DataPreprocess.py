import numpy as np


def parse_data(file_name):
    """
    It parses data from file
    :param file_name: Path to file with a instantion of data
    :return: Numpy array with data
    """
    data = []
    with open(file_name) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            coordinates = line.split()
            data.append((int(coordinates[0]), int(coordinates[1])))
            i += 1

    return np.array(data)


def create_dist_function(data, dist_fun):
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


def create_clusters_from_tree(data, result):
    clusters = np.ones(len(data), dtype=np.int32) * (-1)
    result_tree = np.copy(result)

# dividing points from list of edges to clusters. Each edge consists of indexes of two points it connects.
    out = []
    while len(result_tree) > 0:
        first, *rest = result_tree
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(first)
        result_tree = rest

# assigning each point a number of group it belongs to
    for index, cl in enumerate(out):
        for i, edge in enumerate(data):
            if i in cl:
                clusters[i] = index

# single-point clusters where not taken into account so far - they also need to be assigned a group number
    for i, item in enumerate(clusters):
        if item == -1:
            clusters_number = np.max(clusters)
            clusters[i] = clusters_number+1
    return clusters

