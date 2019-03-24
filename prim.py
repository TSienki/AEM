import numpy as np


def prim(dist_matrix, starting_point):
    """
    :param dist_matrix:
    :param starting_point: index of starting point
    :return: MST
    """
    matrix = np.copy(dist_matrix)
    n_vertices = matrix.shape[0]
    spanning_edges = []

    visited_vertices = [starting_point]
    num_visited = 1

    diag_indices = np.arange(n_vertices)
    matrix[diag_indices, diag_indices] = np.inf
    while num_visited != n_vertices:
        new_edge = np.argmin(matrix[visited_vertices], axis=None)

        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]

        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])

        matrix[visited_vertices, new_edge[1]] = np.inf
        matrix[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)


def prim_n_tree_generate(dist_matrix, starting_points):
    """
    :param dist_matrix:
    :param starting_points: list of indices of n starting points, for n trees
    :return: MST
    """
    matrix = np.copy(dist_matrix)
    n_vertices = matrix.shape[0]
    spanning_edges = []

    visited_vertices = [*starting_points]
    num_visited = len(visited_vertices)

    diag_indices = np.arange(n_vertices)
    for x in starting_points:
        for y in starting_points:
            if x != y:
                matrix[x, y] = np.inf
    matrix[diag_indices, diag_indices] = np.inf
    while num_visited != n_vertices:
        new_edge = np.argmin(matrix[visited_vertices], axis=None)
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]

        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])

        matrix[visited_vertices, new_edge[1]] = np.inf
        matrix[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)
