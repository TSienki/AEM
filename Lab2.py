import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import draw_scatter
from Algorithms.lab2 import random_groups, greedy_algorithm


def run():
    neighbourhood = 10 #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    clusters_indices = random_groups(data.shape[0])
    # greedy_algorithm(clusters_indices, dist_matrix, neighbourhood)
    draw_scatter(data, clusters_indices)

if "__main__" == __name__:
    run()
