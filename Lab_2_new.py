import time

from Algorithms.new_algorithms import run_algorithm_greedy, run_algorithm_steepest, random_groups, cost_function
from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import draw_scatter


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


def run_measurements(data, dist_matrix, neighbourhood_radius, steps_for_time_measurements=50, option="prim"):
    greedy_times_measurements = []
    steepest_times_measurements = []
    greedy_measurements = []
    steepest_measurements = []
    dist = np.copy(dist_matrix)

    for i in range(steps_for_time_measurements):
        print(i, "started")
        clusters = np.ones(len(data), dtype=np.int32) * (-1)
        clusters = random_groups(data.shape[0])
        greedy_clusters = np.copy(clusters)
        measurement = time_measure(run_algorithm_greedy, (greedy_clusters, dist, neighbourhood_radius))
        greedy_times_measurements.append(measurement)
        greedy_measurements.append(cost_function(dist_matrix, greedy_clusters)[0])

        measurement = time_measure(run_algorithm_steepest, (clusters, dist, neighbourhood_radius))
        steepest_times_measurements.append(measurement)
        steepest_measurements.append(cost_function(dist_matrix, clusters)[0])

        print(i, "finished")

    print(f"Greedy cost min:{min(greedy_measurements)}, max:{max(greedy_measurements)}, avg: {sum(greedy_measurements) / len(greedy_measurements)}")
    print(f"Greedy Time min:{min(greedy_times_measurements)}, max:{max(greedy_times_measurements)}, avg: {sum(greedy_times_measurements) / len(greedy_times_measurements)}")

    print(f"Steepest cost min:{min(steepest_measurements)}, max:{max(steepest_measurements)}, avg: {sum(steepest_measurements) / len(steepest_measurements)}")
    print(f"Steepest Time min:{min(steepest_times_measurements)}, max:{max(steepest_times_measurements)}, avg: {sum(steepest_times_measurements) / len(steepest_times_measurements)}")


def run():
    neighbourhood = 50  #radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    # run_measurements(data, dist_matrix, neighbourhood, 1, "prim")
    # run_measurements(data, dist_matrix, neighbourhood, 1, "kruskal")
    # run_measurements(data, dist_matrix, neighbourhood, 1, "random")
    run_measurements(data, dist_matrix, neighbourhood, 1, "random",)


if "__main__" == __name__:
    run()
