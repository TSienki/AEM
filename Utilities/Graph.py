import matplotlib.pyplot as plt
import numpy as np


def draw_scatter(points):
    """
    It generates basic scatter plot from points
    :param points: points to show
    """
    plt.scatter(points.T[0], points.T[1])
    plt.show()
