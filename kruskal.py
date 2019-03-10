# determining whether there is no cycle
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# union of tree ranks
def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def kruskal(dist_matrix):
    n_vertices = dist_matrix.size
    graph = []
    spanning_edges = []
    num_visited = 0
    parent = []
    rank = []

    # Write matrix elements with indexes to a graph so that the elements could be sorted
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            graph.append([i, j, dist_matrix[i][j]])

    # Sorting edges in increasing weigh
    graph = sorted(graph, key=lambda item: item[2])

    # Create subsets with single elements
    for node in range(n_vertices):
        parent.append(node)
        rank.append(0)

    while num_visited < n_vertices - 1:
        # From the smallest edge incremented with i
        u, v, w = graph[num_visited]
        num_visited = num_visited + 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            spanning_edges.append([u, v, w])
            union(parent, rank, x, y)

    # return all but 9 longest edges
    return spanning_edges[:len(spanning_edges)-9]
