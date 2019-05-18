import time

from Algorithms.lab4 import msls, ils
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
                          neighbourhood=50, candidates=False, cache=True):
    msls_times_measurements = []
    msls_measurements = []
    dist = np.copy(dist_matrix)
    best_clusters = None
    ils_best_clusters = None
    clusters_before_best = None
    ils_clusters_before_best = None
    ils_times_measurements = []
    ils_measurements = []

    ils_big_best_clusters = None
    ils_big_clusters_before_best = None
    ils_big_times_measurements = []
    ils_big_measurements = []

    for i in range(steps_for_time_measurements):
        print(f"Aktualna iteracja to {steps_for_time_measurements}")
        time_measurement, ret = time_measure(msls, (dist, neighbourhood, data, candidates, cache, option))
        cost, best_clusters, clusters_before_best = ret
        msls_times_measurements.append(time_measurement)
        msls_measurements.append(cost)

        ils_time_limit = sum(msls_times_measurements) / len(msls_measurements)

        ils_time_measurement, ret = \
            time_measure(ils, (dist, neighbourhood, data, ils_time_limit, candidates, cache, option, "small"))
        ils_cost, ils_best_clusters, ils_clusters_before_best = ret
        ils_times_measurements.append(ils_time_measurement)
        ils_measurements.append(ils_cost)

        ils_big_time_measurement, big_ret = \
            time_measure(ils, (dist, neighbourhood, data, ils_time_limit, candidates, cache, option, "big"))
        ils_big_cost, ils_big_best_clusters, ils_big_clusters_before_best = big_ret
        ils_big_times_measurements.append(ils_big_time_measurement)
        ils_big_measurements.append(ils_big_cost)

    print(f"MSLS COST min:{min(msls_measurements)}, max:{max(msls_measurements)}, avg: {sum(msls_measurements) / len(msls_measurements)}")
    print(f"MSLS TIME min:{min(msls_times_measurements)}, max:{max(msls_times_measurements)}, avg: {sum(msls_times_measurements) / len(msls_times_measurements)}")
    print(f"ILS with small perturbations COST min:{min(ils_measurements)}, max:{max(ils_measurements)}, avg: {sum(ils_measurements) / len(ils_measurements)}")
    print(f"ILS with small perturbations TIME min:{min(ils_times_measurements)}, max:{max(ils_times_measurements)}, avg: {sum(ils_times_measurements) / len(ils_times_measurements)}")
    print(f"ILS with big perturbations COST min:{min(ils_big_measurements)}, max:{max(ils_big_measurements)}, avg: {sum(ils_big_measurements) / len(ils_big_measurements)}")
    print(f"ILS with big perturbations TIME min:{min(ils_big_times_measurements)}, max:{max(ils_big_times_measurements)}, avg: {sum(ils_big_times_measurements) / len(ils_big_times_measurements)}")
    draw_scatter(data, best_clusters, True)
    draw_scatter(data, clusters_before_best, False)
    draw_scatter(data, ils_best_clusters, True)
    draw_scatter(data, ils_clusters_before_best, False)
    draw_scatter(data, ils_big_best_clusters, True)
    draw_scatter(data, ils_big_clusters_before_best, False)


def run():
    neighbourhood = 50  # radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    print("Random")
    run_measurements_msls(data, dist_matrix, neighbourhood, 10, "random")


if "__main__" == __name__:
    run()