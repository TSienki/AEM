import matplotlib.pyplot as plt
import numpy as np


def draw_scatter(points, clusters):
    """
    It generates basic scatter plot from points
    :param points: points to show
    :param clusters: list of pairs cluster number and data number
    """
    number_clusters = np.max(clusters[:, 0])
    for i in range(number_clusters + 1):
        np_group = clusters[clusters[:, 0] == i]
        # for indices_1 in cluster:
        #     if indices_1 != cluster[0]:
        #         plt.plot((points[cluster[0]][0], points[indices_1][0]),
        #                  (points[cluster[0]][1], points[indices_1][1]))
        plt.scatter(points[np_group].T[0], points[np_group].T[1])
        # One color one cluster
    plt.show()


