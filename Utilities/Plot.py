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
        """
        plt.scatter(points.T[0], points.T[1])
        for group in groups:
            first_index_in_group = group[0]
            for index in group:
                if index != first_index_in_group:
                    plt.plot((points[first_index_in_group][0], points[index][0]),
                             (points[first_index_in_group][1], points[index][1]))
        plt.show()
        # ax = self.figure.add_subplot("111")
        # ax.scatter(points.T[0], points.T[1])

    def draw_lines(self, points, group):
        pass

    def show(self):
        pass
        # fig = self.figure
        # print("elo")
        # fig = plt.Figure()
        # test = fig.add_subplot("111")
        # plt.scatter(np.array([0, 1, 2]), np.array([4, 5, 6]))
        # self.ax.set_limx(200)
        # fig.draw()
        # plt.show()
        # plt.show()
        # self.figure.savefig("xf.png")
