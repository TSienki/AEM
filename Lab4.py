import time

from Algorithms.lab4 import msls
from Utilities.DataPreprocess import parse_data, create_dist_function, create_clusters_from_tree
from Utilities.Plot import draw_scatter
from Lab1 import create_n_trees_kruskal, create_n_trees_prim


import numpy as np


def time_measure(func, args_for_func):
    """
    :param func:
    :param args_for_func:
    :return: time in seconds
    """
    start = time.time()
    ret = func(*args_for_func)
    end = time.time()
    return end - start, ret


def run_measurements_msls(data, dist_matrix, neighbourhood_radius, steps_for_time_measurements=1,  option="random",
                          neighbourhood=50, candidates=False, cache=False):
    msls_times_measurements = []
    msls_measurements = []
    dist = np.copy(dist_matrix)
    best_clusters_msls = None
    clusters_before_msls = None

    for i in range(steps_for_time_measurements):
        time_measurement, ret = time_measure(msls, (dist, neighbourhood, data, candidates, cache, option))
        cost, best_clusters, cluster_before_best = ret
        msls_times_measurements.append(time_measurement)
        msls_measurements.append(cost)

    print(f"Msls without cache and candidates cost min:{min(msls_measurements)}, max:{max(msls_measurements)}, avg: {sum(msls_measurements) / len(msls_measurements)}")
    print(f"Msls without cache and candidates Time min:{min(msls_times_measurements)}, max:{max(msls_times_measurements)}, avg: {sum(msls_times_measurements) / len(msls_times_measurements)}")

    draw_scatter(data, best_clusters_msls, True)
    draw_scatter(data, clusters_before_msls, False)


def run():
    neighbourhood = 50  # radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    # print("Prim")
    # run_measurements(data, dist_matrix, neighbourhood, 1, "prim")
    # print("Kruskal")
    # run_measurements(data, dist_matrix, neighbourhood, 1, "kruskal")
    print("Random")
    # run_measurements(data, dist_matrix, neighbourhood, 1, "random")
    run_measurements_msls(data, dist_matrix, neighbourhood, 1, "random")


if "__main__" == __name__:
    run()