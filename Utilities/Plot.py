import matplotlib.pyplot as plt
import numpy as np


class Plot:
    # TODO: Try to make nice class to show plots

    def __init__(self):
        self.figure = plt.Figure()

    def draw_scatter(self, points, groups):
        """
        It generates basic scatter plot from points
        :param points: points to show
        :param groups: points indices of each group
        """
        for group in groups:
            np_group = np.array(group)
            plt.scatter(points[np_group].T[0], points[np_group].T[1])
        plt.show()

    def draw_lines(self, points, group):
        pass

    def show(self):
        pass
        # fig = self.figure
        # fig = plt.Figure()
        # test = fig.add_subplot("111")
        # plt.scatter(np.array([0, 1, 2]), np.array([4, 5, 6]))
        # self.ax.set_limx(200)
        # fig.draw()
        # plt.show()
        # plt.show()
        # self.figure.savefig("xf.png")
