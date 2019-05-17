from Utilities.DataPreprocess import parse_data, create_dist_function
from Utilities.Plot import draw_similarity
from Algorithms.lab5 import random_groups, cost_function
from Algorithms.lab2_new import run_algorithm_greedy
import numpy as np
import itertools
from scipy.stats import pearsonr


def count_similarity(optimums, optimums_costs):
    best_optimum = np.argmin(optimums_costs)
    best_optimum_cluster = optimums[best_optimum]
    similarities = np.ones_like(optimums_costs)
    avg_similarities = np.ones_like(optimums_costs)
    for i in range(len(optimums_costs)):
        similarity = 0
        for pair in itertools.combinations(range(len(optimums[0])), 2):
            if best_optimum_cluster[pair[0]] == best_optimum_cluster[pair[1]]:
                if optimums[i][pair[0]] == optimums[i][pair[1]]:
                    similarity += 1
        similarities[i] = similarity
        other_similarities = np.zeros_like(optimums_costs)
        for j in range(len(optimums_costs)):
            if i < j:
                if j != best_optimum:
                    sim = 0
                    for pair in itertools.combinations(range(len(optimums[0])), 2):
                        if optimums[j][pair[0]] == optimums[j][pair[1]]:
                            if optimums[i][pair[0]] == optimums[i][pair[1]]:
                                sim += 1
                    other_similarities[j] = sim
        avg_similarities[i] = sum(i for i in other_similarities) / (len(other_similarities) - 1)
    return similarities, avg_similarities


def run_measurements_msls(data, dist_matrix, neighbourhood_radius, steps=500):
    local_optimums = []
    local_optimums_costs = []

    for i in range(steps):
        clusters = random_groups(data.shape[0])
        run_algorithm_greedy(clusters, dist_matrix, neighbourhood_radius)
        local_optimums.append(clusters)
        local_optimums_costs.append(cost_function(dist_matrix, clusters)[0])
    similarities, avg_similarities = count_similarity(local_optimums, local_optimums_costs)
    draw_similarity(similarities, avg_similarities, local_optimums_costs)
    cor_coef = pearsonr(local_optimums_costs, similarities)[0]
    cor_coef2 = pearsonr(local_optimums_costs, avg_similarities)[0]
    print(f"Współczynnik koleracji podobieństwa względem najlepszego optimum lokalnego: {cor_coef}")
    print(f"Współczynnik koleracji średniego podobieństwa względem pozostałych optimów lokalnych: {cor_coef2}")


def run():
    neighbourhood = 50  # radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    run_measurements_msls(data, dist_matrix, neighbourhood, 500)


if "__main__" == __name__:
    run()
