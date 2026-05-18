# data_structures/graph.py
from collections import defaultdict, deque
import heapq


class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = defaultdict(list)

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []

    def neighbors(self, v):
        return [node for node, _ in self.adj[v]]

    def vertices(self):
        return list(self.adj.keys())

    def bfs(self, start):
        visited = set()
        queue   = deque([start])
        order   = []
        visited.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def dfs(self, start):
        visited = set()
        order   = []
        self._dfs(start, visited, order)
        return order

    def _dfs(self, node, visited, order):
        visited.add(node)
        order.append(node)
        for neighbor in self.neighbors(node):
            if neighbor not in visited:
                self._dfs(neighbor, visited, order)

    def dfs_iterative(self, start):
        visited = set()
        stack   = [start]
        order   = []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                order.append(node)
                for neighbor in reversed(self.neighbors(node)):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return order

    def dijkstra(self, start):
        dist = {v: float("inf") for v in self.adj}
        dist[start] = 0
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def has_cycle_undirected(self):
        visited = set()
        def dfs(v, parent):
            visited.add(v)
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    if dfs(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False
        for v in self.adj:
            if v not in visited:
                if dfs(v, -1):
                    return True
        return False

    def topological_sort(self):
        visited = set()
        stack   = []
        def dfs(v):
            visited.add(v)
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(v)
        for v in self.adj:
            if v not in visited:
                dfs(v)
        return stack[::-1]

    def connected_components(self):
        visited    = set()
        components = []
        for v in self.adj:
            if v not in visited:
                component = []
                queue = deque([v])
                visited.add(v)
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in self.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        return components