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
