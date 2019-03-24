import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import draw_scatter
from Algorithms.lab2 import random_groups, run_algorithm


def run():
    neighbourhood = 50  #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    clusters = random_groups(data.shape[0])

    clusters_2 = np.copy(clusters)
    dist_2 = np.copy(dist_matrix)

    draw_scatter(data, clusters, True)

    print("Greedy")
    run_algorithm(clusters, dist_matrix, neighbourhood, algorithm="greedy")
    draw_scatter(data, clusters, True)

    print("Steepest")
    run_algorithm(clusters_2, dist_2, neighbourhood, algorithm="steepest")
    draw_scatter(data, clusters_2, True)

if "__main__" == __name__:
    run()
