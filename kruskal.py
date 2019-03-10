class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []

    # function to add an edge to graph
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # determining whether there is no cycle
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # union of tree ranks
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal_mst(self):

        result = []

        i = 0
        e = 0

        # Sorting edges in increasing weigh
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        # Create subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            # From the smallest edge incremented with i
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            e = e + 1
            if x != y:
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        return result
