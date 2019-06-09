import time

from scipy.stats import pearsonr

from Utilities.DataPreprocess import parse_data, create_dist_function
from Algorithms.lab4 import ils
from Algorithms.lab5 import random_groups
from Algorithms.lab6 import msls, evolutionary
import numpy as np
import itertools

from Utilities.Plot import draw_scatter, draw_similarity

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


def run_measurements_msls(data, dist_matrix, neighbourhood=50, steps_for_time_measurements=1,  option="random",
                           candidates=False, cache=True):
    dist = np.copy(dist_matrix)
    msls_times_measurements = []
    msls_costs = []
    ils_big_costs = []
    ils_small_costs = []
    evo_costs = []

    for i in range(steps_for_time_measurements):
        print(f"CURRENT ITERATION: {i}")
        clusters = random_groups(data.shape[0])
        print("MSLS...")
        time_limit, ret = time_measure(msls, (dist, neighbourhood, data, candidates, cache, option))
        msls_times_measurements.append(time_limit)
        local_optimums = ret[0]
        local_optimums_costs = ret[1]
        msls_costs.append(np.max(local_optimums_costs))

        print("ILS with small perturbation...")
        ret_small = ils(dist, neighbourhood, data, time_limit, candidates, cache, option, "small")
        ils_small_costs.append(ret_small[0])

        print("ILS with big perturbation...")
        ret_big = ils(dist, neighbourhood, data, time_limit, candidates, cache, option, "big")
        ils_big_costs.append(ret_big[0])

        print("Evolution algorithm...")
        ret_evo = evolutionary(dist, neighbourhood, local_optimums, local_optimums_costs, time_limit)
        evo_costs.append(np.max(ret_evo[1]))

        # similarities, avg_similarities = count_similarity(local_optimums, local_optimums_costs)
        # draw_similarity(similarities, avg_similarities, local_optimums_costs)
        # cor_coef = pearsonr(local_optimums_costs, similarities)[0]
        # cor_coef2 = pearsonr(local_optimums_costs, avg_similarities)[0]
        # print(f"Współczynnik koleracji podobieństwa względem najlepszego optimum lokalnego: {cor_coef}")
        # print(f"Współczynnik koleracji średniego podobieństwa względem pozostałych optimów lokalnych: {cor_coef2}")

    print(f"MSLS COST min:{min(msls_costs)}, max:{max(msls_costs)}, avg: {sum(msls_costs) / len(msls_costs)}")
    print(f"ILS with small perturbations COST min:{min(ils_small_costs)}, max:{max(ils_small_costs)}, avg: "
          f"{sum(ils_small_costs) / len(ils_small_costs)}")
    print(f"ILS with big perturbations COST min:{min(ils_big_costs)}, max:"
          f"{max(ils_big_costs)}, avg: {sum(ils_big_costs) / len(ils_big_costs)}")
    print(f"Evolutionary algorithm COST min:{min(evo_costs)}, max:"
          f"{max(evo_costs)}, avg: {sum(evo_costs) / len(evo_costs)}")
    draw_scatter(data, ret_evo[0][np.argmax(ret_evo[1])], True)


def run():
    neighbourhood = 50  # radius of neighbourhood
    data = parse_data("data/objects20_06.data")
    dist_matrix = create_dist_function(data, lambda x1, x2: np.linalg.norm(x1 - x2))
    run_measurements_msls(data, dist_matrix, neighbourhood, 2, "random")


if "__main__" == __name__:
    run()