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


# ======================= 5 =======================

# ======================= 6 =======================
def generate_random_graph(num_nodes: int, density: float = 0.25):
    """
    Генерує випадковий граф та повертає його у двох представленнях:
    матриця суміжності та словник суміжності.
    :param num_nodes: Кількість вершин
    :param density: Щільність графа (від 0.0 до 1.0). 0.25 - це розріджений граф.
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
    Compare speed of DFS/BFS for Adjacency Matrix vs Adjacency Dict
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
