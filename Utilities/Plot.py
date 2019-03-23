import matplotlib.pyplot as plt
import numpy as np


def draw_scatter(points, clusters, with_lines):
    """
    It generates basic scatter plot from points
    :param points: points to show
    :param clusters: list of pairs cluster number and data number
    """
    number_clusters = np.max(clusters)
    for i in range(number_clusters + 1):
        np_cluster = np.argwhere(clusters == i)
        if with_lines:
            for indices_1 in np_cluster:
                if indices_1 != np_cluster[0, 0]:
                    plt.plot((points[np_cluster[0, 0]][0], points[indices_1][0][0]),
                             (points[np_cluster[0, 0]][1], points[indices_1][0][1]))
        plt.scatter(points[np_cluster].T[0], points[np_cluster].T[1])
    plt.show()


