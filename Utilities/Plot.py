import matplotlib.pyplot as plt
import numpy as np


def draw_scatter(points, clusters):
    """
    It generates basic scatter plot from points
    :param points: points to show
    :param clusters: list of pairs cluster number and data number
    """
    number_clusters = np.max(clusters)
    for i in range(number_clusters + 1):
        np_group = np.argwhere(clusters == i)
        plt.scatter(points[np_group].T[0], points[np_group].T[1])
    plt.show()


