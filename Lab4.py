import time

from Algorithms.lab3_new import run_algorithm_steepest, random_groups, cost_function
from Utilities.DataPreprocess import parse_data, create_dist_function, create_clusters_from_tree
from Utilities.Plot import draw_scatter
from Lab1 import create_n_trees_kruskal, create_n_trees_prim
from Algorithms.lab4 import get_neighbourhood
import numpy as np


def make_perturbations(clusters, number_of_perturbations, neighbourhood_radius, dist_matrix):
    for i in range(number_of_perturbations):
        point_to_perturbate = np.random.choice(len(clusters), 1)
        point_neighbourhood = get_neighbourhood(clusters, dist_matrix, neighbourhood_radius, point_to_perturbate, False)
        second_point = np.random.choice([i for i in range(len(clusters)) if i not in point_neighbourhood], 1)
        temp = clusters[second_point]
        clusters[second_point] = clusters[point_to_perturbate]
        clusters[point_to_perturbate] = temp
    return clusters


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
                     option="prim", candidates = False, cache=False, perturbations=False):
    steepest_times_measurements = []
    steepest_measurements = []
    dist = np.copy(dist_matrix)

    best_steepest = np.inf
    best_clusters_steepest = []
    costs_steepest = []
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

        for i in range(10):
            steepest_clusters = np.copy(clusters)
            steepest_clusters_with_candidates = np.copy(clusters)

            measurement = time_measure(run_algorithm_steepest, (steepest_clusters, dist, neighbourhood_radius, candidates, cache))
            steepest_times_measurements.append(measurement)
            steepest_cost = cost_function(dist_matrix, steepest_clusters)[0]
            steepest_measurements.append(steepest_cost)
            costs_steepest.append(steepest_cost)
            if steepest_cost < best_steepest:
                best_steepest = steepest_cost
                best_clusters_steepest = steepest_clusters
                clusters_before_steepest = clusters
            if perturbations:
                make_perturbations(clusters, 20, 50, dist_matrix)

    print(f"Steepest cost min:{min(steepest_measurements)}, max:{max(steepest_measurements)}, avg: {sum(steepest_measurements) / len(steepest_measurements)}")
    print(f"Steepest Time min:{min(steepest_times_measurements)}, max:{max(steepest_times_measurements)}, avg: {sum(steepest_times_measurements) / len(steepest_times_measurements)}")

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
    run_measurements(data, dist_matrix, neighbourhood, 1, "random", candidates=True, cache=False, perturbations=True)


if "__main__" == __name__:
    run()
