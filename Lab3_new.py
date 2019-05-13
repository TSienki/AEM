import time

from Algorithms.lab3_new import run_algorithm_steepest, random_groups, cost_function
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
    func(*args_for_func)
    end = time.time()
    return end - start


def run_measurements(data, dist_matrix, neighbourhood_radius, steps_for_time_measurements=50,
                     option="prim"):
    steepest_times_measurements = []
    steepest_measurements = []
    dist = np.copy(dist_matrix)

    cache_times_measurements = []
    cache_measurements = []

    candidates_times_measurements = []
    candidates_measurements = []

    best_steepest = np.inf
    best_clusters_steepest = None

    best_cache = np.inf
    best_clusters_cache = None

    best_candidates = np.inf
    best_candidates_clusters = None

    clusters_before_steepest = None
    clusters_before_cache = None
    clusters_before_candidates = None

    for i in range(steps_for_time_measurements):
        clusters = np.ones(len(data), dtype=np.int32) * (-1)

        if option == "prim":
            result = create_n_trees_prim(data, dist_matrix, 20, 1)
            clusters = create_clusters_from_tree(data, result)
        elif option == "kruskal":
            result = create_n_trees_kruskal(data, dist_matrix, 20, 1)
            clusters = create_clusters_from_tree(data, result)
        elif option == "random":
            clusters = random_groups(data.shape[0])

        steepest_clusters = np.copy(clusters)
        cache_cluster = np.copy(clusters)
        candidates_cluster = np.copy(clusters)

        measurement = time_measure(run_algorithm_steepest, (steepest_clusters, dist, neighbourhood_radius, False, False))
        steepest_times_measurements.append(measurement)
        steepest_cost = cost_function(dist_matrix, steepest_clusters)[0]
        steepest_measurements.append(steepest_cost)

        measurement = time_measure(run_algorithm_steepest, (cache_cluster, dist, neighbourhood_radius, False, True))
        cache_times_measurements.append(measurement)
        cache_cost = cost_function(dist_matrix, steepest_clusters)[0]
        cache_measurements.append(cache_cost)

        measurement = time_measure(run_algorithm_steepest, (candidates_cluster, dist, neighbourhood_radius, True, False))
        candidates_times_measurements.append(measurement)
        candidates_cost = cost_function(dist_matrix, candidates_cluster)[0]
        candidates_measurements.append(cache_cost)

        if steepest_cost < best_steepest:
            best_steepest = steepest_cost
            best_clusters_steepest = steepest_clusters
            clusters_before_steepest = clusters

        if cache_cost < best_cache:
            best_cache = cache_cost
            best_clusters_cache = cache_cluster
            clusters_before_cache = clusters

        if candidates_cost < best_candidates:
            best_candidates = candidates_cost
            best_candidates_clusters = candidates_cluster
            clusters_before_candidates = clusters

    print(f"Steepest cost min:{min(steepest_measurements)}, max:{max(steepest_measurements)}, avg: {sum(steepest_measurements) / len(steepest_measurements)}")
    print(f"Steepest Time min:{min(steepest_times_measurements)}, max:{max(steepest_times_measurements)}, avg: {sum(steepest_times_measurements) / len(steepest_times_measurements)}")

    print(f"Cache steepest cost min:{min(cache_measurements)}, max:{max(cache_measurements)}, avg: {sum(cache_measurements) / len(cache_measurements)}")
    print(f"Cache steepest Time min:{min(cache_times_measurements)}, max:{max(cache_times_measurements)}, avg: {sum(cache_times_measurements) / len(cache_times_measurements)}")

    print(f"Candidates steepest cost min:{min(candidates_measurements)}, max:{max(candidates_measurements)}, avg: {sum(candidates_measurements) / len(candidates_measurements)}")
    print(f"Candidates steepest Time min:{min(candidates_times_measurements)}, max:{max(candidates_times_measurements)}, avg: {sum(candidates_times_measurements) / len(candidates_times_measurements)}")

    draw_scatter(data, best_clusters_steepest, True)
    draw_scatter(data, clusters_before_steepest, False)


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
    run_measurements(data, dist_matrix, neighbourhood, 10, "random")


if "__main__" == __name__:
    run()
