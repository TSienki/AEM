import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm



def draw_scatter(points, clusters, with_lines):
    """
    It generates basic scatter plot from points
    :param points: points to show
    :param clusters: list of pairs cluster number and data number
    """
    number_clusters = np.max(clusters)
    colors = cm.rainbow(np.linspace(0, 1, number_clusters+1))
    for i in range(number_clusters + 1):
        np_cluster = np.argwhere(clusters == i)
        if with_lines:
            for indices_1 in np_cluster:
                if indices_1 != np_cluster[0, 0]:
                    plt.plot((points[np_cluster[0, 0]][0], points[indices_1][0][0]),
                             (points[np_cluster[0, 0]][1], points[indices_1][0][1]), color=colors[i])
        plt.scatter(points[np_cluster].T[0], points[np_cluster].T[1], color=colors[i])
    plt.show()


def draw_similarity(similarities, avg_similarities, costs):
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.scatter(costs, similarities)
    ax1.set_ylim(bottom=0)
    ax1.set_title("Podobieństwo względem najlepszego optimum lokalnego")
    ax1.set_xlabel("Wartość funkcji celu")
    ax1.set_ylabel("Podobieństwo")

    ax2.scatter(costs, avg_similarities)
    ax2.set_ylim(bottom=0)
    ax2.set_title("Średnie podobieństwo względem pozostałych optimów lokalnych")
    ax2.set_xlabel("Wartość funkcji celu")
    ax2.set_ylabel("Podobieństwo")

    plt.show()


