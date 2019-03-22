import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import Plot
from Algorithms.lab2 import random_groups


def run():
    plot = Plot()
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    # print(dist_matrix)
    group_indices = random_groups(data.shape[0])
    plot.draw_scatter(data, group_indices)
    # plot.draw_lines(data, group_indices)
    # plot.show()


if "__main__" == __name__:
    run()
