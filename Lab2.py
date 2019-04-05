import time
import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function, create_clusters_from_tree
from Utilities.Plot import draw_scatter
from Algorithms.lab2 import random_groups, run_algorithm, count_costs
from Lab1 import create_n_trees_kruskal, create_n_trees_prim



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


def run_measurements(data, dist_matrix, neighbourhood, steps_for_time_measurements=1, option="prim"):
    times_measurements = []
    times_measurements_2 = []
    dist_1 = np.copy(dist_matrix)
    dist_2 = np.copy(dist_matrix)
    best_greedy = 10000
    best_clusters_greedy = []
    best_steepest = 10000
    best_clusters_steepest = []
    costs_greedy = []
    costs_steepest = []
    clusters_before_greedy = []
    clusters_before_steepest = []

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
        clusters_2 = np.copy(clusters)
        clusters_before = np.copy(clusters)
        measurement = time_measure(run_algorithm, (clusters, dist_1, neighbourhood, "greedy"))
        times_measurements.append(measurement)
        measurement2 = time_measure(run_algorithm, (clusters_2, dist_2, neighbourhood, "steepest"))
        times_measurements_2.append(measurement2)
        cost = sum(count_costs(clusters, dist_1, 20))/20
        cost2 = sum(count_costs(clusters_2, dist_2, 20))/20
        costs_greedy.append(cost)
        costs_steepest.append(cost2)
        if cost < best_greedy:
            best_greedy = cost
            best_clusters_greedy = clusters
            clusters_before_greedy = clusters_before
        if cost2 < best_steepest:
            best_steepest = cost2
            best_clusters_steepest = clusters_2
            clusters_before_steepest = clusters_before
    print(option)
    print(np.max(best_clusters_steepest))
    print(np.max(best_clusters_greedy))
    print(f"Najmniejszy koszt dla lokalnego przeszukiwania w wersji zachłannej dla wstępnych danych {option} wynosi {min(costs_greedy)}, "
          f"największy {max(costs_greedy)}, średni {sum(costs_greedy)/len(costs_greedy)}.\n"
          f"Najmniejszt koszt dla lokalnego przeszukiwania w wersji stromej wynosi {min(costs_steepest)}, "
          f"największy {max(costs_steepest)}, średni {sum(costs_steepest)/len(costs_steepest)}.")
    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków dla algorytmu greedy to "
          f"min: {min(times_measurements)} sekund, max: {max(times_measurements)} sekund i "
          f"avg: {sum(times_measurements) / len(times_measurements)} sekund")
    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków dla algorytmu steepest to "
          f"min: {min(times_measurements_2)} sekund, max: {max(times_measurements_2)} sekund i "
          f"avg: {sum(times_measurements_2) / len(times_measurements_2)} sekund")
    draw_scatter(data, best_clusters_greedy, True)
    draw_scatter(data, best_clusters_steepest, True)
    draw_scatter(data, clusters_before_greedy, False)
    draw_scatter(data, clusters_before_steepest, False)


def run():
    neighbourhood = 40  #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    # run_measurements(data, dist_matrix, neighbourhood, 100, "prim")
    # run_measurements(data, dist_matrix, neighbourhood, 100, "kruskal")
    run_measurements(data, dist_matrix, neighbourhood, 1, "random")


if "__main__" == __name__:
    run()
