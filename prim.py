import numpy as np


def prim(dist_matrix, starting_point):
    matrix = np.copy(dist_matrix)
    n_vertices = matrix.shape[0]
    spanning_edges = []

    visited_vertices = [starting_point]
    num_visited = 1

    diag_indices = np.arange(n_vertices)
    matrix[diag_indices, diag_indices] = np.inf

    while num_visited != n_vertices:
        new_edge = np.argmin(matrix[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        matrix[visited_vertices, new_edge[1]] = np.inf
        matrix[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)
