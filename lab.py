"""
Lab 2 template
"""


# ======================= 1 =======================
def read_incidence_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the incidence matrix of a given graph
    >>> read_incidence_matrix('input.dot')
    [[-1, -1, 1, 0, 1, 0], [1, 0, -1, -1, 0, 1], [0, 1, 0, 1, -1, -1]]
    """
    pairs = []
    vertices = set()
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if '->' in line:
                a, b = line.strip()[:-1].split(' -> ')
                a, b = int(a), int(b)
                vertices.update([a, b])
                pairs.append((a, b))

        m = len(pairs)
        n = len(vertices)
        matrix = [[0] * m for _ in range(n)]
        for i, (a, b) in enumerate(pairs):
            matrix[a][i] = -1
            matrix[b][i] = 1
    return matrix


def read_adjacency_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the adjacency matrix of a given graph
    >>> read_adjacency_matrix('input.dot')
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    """
    pairs = []
    vertices = set()
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if '->' in line:
                a, b = line.strip()[:-1].split(' -> ')
                a, b = int(a), int(b)
                vertices.update([a, b])
                pairs.append((a, b))

        size = max(vertices) + 1
        matrix = [[0] * size for _ in range(size)]
        for a, b in pairs:
            matrix[a][b] = 1
    return matrix


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict[int, list[int]]: the adjacency dict of a given graph
    >>> read_adjacency_dict('input.dot')
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    matrix = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if '->' in line:
                a, b = line.strip()[:-1].split(' -> ')
                a, b = int(a), int(b)
                if a not in matrix:
                    matrix[a] = []
                matrix[a].append(b)
    return matrix


# ======================= 2 =======================
def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """

    visited = []

    def dfs(current_node):
        neighbors = sorted(graph.get(current_node, []))
        if current_node not in visited:
            visited.append(current_node)
            for n in neighbors:
                dfs(n)

    dfs(start)
    return visited


def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = []

    def matrix_to_dict(graph: list[list[int]]) -> dict[int, list[int]]:
        dict_1 = {}
        for i in range(len(graph)):
            neighbors = []
            for j in range(len(graph[i])):
                if graph[i][j] == 1:
                    neighbors.append(j)
            dict_1[i] = neighbors

        return dict_1

    graph = matrix_to_dict(graph)

    def dfs(current_node):
        neighbors = sorted(graph.get(current_node, []))
        if current_node not in visited:
            visited.append(current_node)
            for n in neighbors:
                dfs(n)

    dfs(start)
    return visited


# ======================= 3 =======================
def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            result.append(v)

            for u in graph[v]:
                if u not in visited:
                    stack.append(u)

    return sorted(result)


def iterative_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    stack = [start]
    result = []
    n = len(graph)

    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            result.append(v)

            for u in range(n):
                if graph[v][u] == 1 and u not in visited:
                    stack.append(u)

    return sorted(result)


def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = {start}
    queue = [start]
    result = []

    while queue:
        v = queue.pop(0)  # беремо перший елемент
        result.append(v)
        for u in graph[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)  # додаємо в кінець

    return result


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = {start}
    queue = [start]
    result = []
    n = len(graph)

    while queue:
        v = queue.pop(0)
        result.append(v)

        for u in range(n):
            if graph[v][u] == 1 and u not in visited:
                visited.add(u)
                queue.append(u)

    return result


# ======================= 4 =======================
def adjacency_matrix_radius(graph: list[list[int]]) -> int:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    1
    """

    def matrix_to_dict(graph: list[list[int]]) -> dict[int, list[int]]:
        dict_1 = {}
        for i in range(len(graph)):
            neighbors = []
            for j in range(len(graph[i])):
                if graph[i][j] == 1:
                    neighbors.append(j)
            dict_1[i] = neighbors
        return dict_1

    graph = matrix_to_dict(graph)

    if not graph:
        return

    def bfs(start: int) -> int:
        distances = {start: 0}
        queue = [start]
        i = 0

        while i < len(queue):
            current = queue[i]
            i += 1

            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return max(distances.values())

    return min(bfs(v) for v in graph)


def adjacency_dict_radius(graph: dict[int, list[int]]) -> int:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1]})
    1
    """
    if not graph:
        return

    def bfs(start: int) -> int:
        distances = {start: 0}
        queue = [start]
        i = 0

        while i < len(queue):
            current = queue[i]
            i += 1

            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return max(distances.values())

    return min(bfs(v) for v in graph)


# ======================= 5,6 =======================
def find_cycles_any(graph: list[list[int]] | dict[int, list[int]]) -> list[list[int]]:
    """
    Finds all cycles
    >>> find_cycles_any(read_adjacency_dict('test2.dot'))
    [[0, 1, 2], [1, 2, 3], [0, 1, 2, 3]
    >>> find_cycles_any(read_incidence_matrix('test1.dot'))
    [[0, 1, 2], [0, 3, 1, 2]]
    >>> find_cycles_any(read_adjacency_matrix('input.dot'))
    [[0, 1, 2]]
    """
    if isinstance(graph, dict): #adjacency_dict
        n = max(graph.keys()) + 1

        def get_neighbors(v):
            return graph.get(v, [])

        return find_cycles(get_neighbors, n)

    if any(-1 in i for i in graph): #incidence_matrix
        n = len(graph)
        m = len(graph[0])

        adj = {i: [] for i in range(n)}

        for e in range(m):
            a = b = None
            for v in range(n):
                if graph[v][e] == -1:
                    a = v
                elif graph[v][e] == 1:
                    b = v
            if a is not None and b is not None:
                adj[a].append(b)

        def get_neighbors(v):
            return adj[v]

        return find_cycles(get_neighbors, n)
    else: #adjacency_matrix
        n = len(graph)

        def get_neighbors(v):
            result = []
            for u in range(n):
                if graph[v][u] == 1:
                    result.append(u)
            return result

        return find_cycles(get_neighbors, n)

def find_cycles(get_neighbors, n: int) -> list[list[int]]:
    cycles = []
    stack = []
    unique = []

    def dfs(v):
        stack.append(v)

        for u in get_neighbors(v):
            if u in stack:  # знайдено цикл
                idx = stack.index(u)
                cycle = stack[idx:].copy()
                if len(cycle) >= 3:
                    indx = cycle.index(min(cycle))
                    default_cycle = cycle[indx:] + cycle[:indx]
                    if set(default_cycle) not in unique:
                        cycles.append(default_cycle)
                        unique.append(set(default_cycle))
            else:
                dfs(u)  # не перевіряємо visited глобально

        stack.pop()

    for start in range(n):
        dfs(start)

    return [list(a) for a in cycles ]

# ======================= 6 =======================
def generate_random_graph(num_nodes: int, density: float = 0.25):
    """
    Generates a random graph and returns it in two representations:
    an adjacency matrix and an adjacency dictionary.
    :param num_nodes: Number of vertices
    :param density: Graph density (from 0.0 to 1.0)
    """
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    dictionary = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                matrix[i][j] = 1
                matrix[j][i] = 1

                dictionary[i].append(j)
                dictionary[j].append(i)

    return matrix, dictionary

def compare_algorithms(n_values: list[int]):
    """
    Benchmarks DFS and BFS performance on Adjacency Matrices vs. Dictionaries.
    Generates random graphs for each size in `n_values`, prints a timing table,
    and plots the execution times using Matplotlib.
    :param n_values: List of node counts to test (e.g., [100, 500, 1000]).
    Returns:
        tuple: Four lists containing execution times for Matrix DFS, Dict DFS,
               Matrix BFS, and Dict BFS respectively.
    """
    matrix_dfs_times = []
    dict_dfs_times = []
    matrix_bfs_times = []
    dict_bfs_times = []

    print(f"{'N':<10} | {'Mat DFS':<10} | {'Dict DFS':<10} | {'Mat BFS':<10} | {'Dict BFS':<10}")
    print("-" * 65)

    for n in n_values:
        matrix, dictionary = generate_random_graph(n, density=0.2)
        start_node = 0

        start_time = time.time()
        iterative_adjacency_dict_dfs(dictionary, start_node)
        dict_dfs_times.append(time.time() - start_time)

        start_time = time.time()
        iterative_adjacency_matrix_dfs(matrix, start_node)
        matrix_dfs_times.append(time.time() - start_time)

        start_time = time.time()
        iterative_adjacency_dict_bfs(dictionary, start_node)
        dict_bfs_times.append(time.time() - start_time)

        start_time = time.time()
        iterative_adjacency_matrix_bfs(matrix, start_node)
        matrix_bfs_times.append(time.time() - start_time)

        print(
            f"{n:<10} | "
            f"{matrix_dfs_times[-1]:.6f}   | "
            f"{dict_dfs_times[-1]:.6f}   | "
            f"{matrix_bfs_times[-1]:.6f}   | "
            f"{dict_bfs_times[-1]:.6f}"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(n_values, matrix_dfs_times, label='Matrix DFS', marker='o')
    plt.plot(n_values, dict_dfs_times, label='Dict DFS', marker='o')
    plt.plot(n_values, matrix_bfs_times, label='Matrix BFS', marker='x', linestyle='--')
    plt.plot(n_values, dict_bfs_times, label='Dict BFS', marker='x', linestyle='--')

    plt.xlabel("Кількість вершин (n)")
    plt.ylabel("Час виконання (с)")
    plt.title("Порівняння ефективності: Матриця vs Словник")
    plt.legend()
    plt.grid(True)

    plt.show()

    return matrix_dfs_times, dict_dfs_times, matrix_bfs_times, dict_bfs_times

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    compare_algorithms([10, 50, 100, 200, 300, 400, 500])
