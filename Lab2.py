import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import draw_scatter
from Algorithms.lab2 import random_groups, greedy_algorithm


def run():
    neighbourhood = 50  #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    clusters = random_groups(data.shape[0])
    draw_scatter(data, clusters, True)
    greedy_algorithm(clusters, dist_matrix, neighbourhood)
    draw_scatter(data, clusters, True)

if "__main__" == __name__:
    run()
