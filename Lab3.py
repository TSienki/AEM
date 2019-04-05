import time
import numpy as np
from Utilities.DataPreprocess import parse_data, create_dist_function, create_clusters_from_tree
from Utilities.Plot import draw_scatter
from Algorithms.lab3 import random_groups, run_algorithm, count_costs
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


def run_measurements(data, dist_matrix, neighbourhood, steps_for_time_measurements=1):
    dist_1 = np.copy(dist_matrix)
    costs_greedy = []
    times_measurements = []
    clusters = None
    # draw_scatter(data, clusters, False)

    for i in range(steps_for_time_measurements):
        clusters = random_groups(data.shape[0])
        measurement = time_measure(run_algorithm, (clusters, dist_1, neighbourhood, "cache"))
        times_measurements.append(measurement)
        cost = sum(count_costs(clusters, dist_1, 20))/20
        print("Koszt dla iteracji " + str(i) + ": " + str(cost))
        costs_greedy.append(cost)
        draw_scatter(data, clusters, True)

    print(f"Najmniejszy koszt dla lokalnego przeszukiwania w wersji zachłannej wynosi {min(costs_greedy)}, "
          f"największy {max(costs_greedy)}, średni {sum(costs_greedy)/len(costs_greedy)}.\n")
    print(f"Pomiary czasu dla {steps_for_time_measurements} kroków dla algorytmu greedy to "
          f"min: {min(times_measurements)} sekund, max: {max(times_measurements)} sekund i "
          f"avg: {sum(times_measurements) / len(times_measurements)} sekund")
    # draw_scatter(data, clusters, True)


def run():
    neighbourhood = 50  #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    run_measurements(data, dist_matrix, neighbourhood, 10)


if "__main__" == __name__:
    run()
