# Algorithms and Data Structures

## Big O Notation

### Common runtimes

| Big O         | name         | change in runtime if I double $n$? |
| ------------- | ------------ | ---------------------------------- |
| $O(1)$        | constant     | same                               |
| $O(\log n)$   | logarithmic  | increased by a constant            |
| $O(\sqrt{n})$ | square root  | increased by roughly 1.4x          |
| $O(n)$        | linear       | 2x                                 |
| $O(n \log n)$ | linearithmic | roughly 2x                         |
| $O(n^2)$      | quadratic    | 4x                                 |
| $O(n^3)$      | cubic        | 8x                                 |
| $O(n^k)$      | polynomial   | increase by a factor of $2^k$      |
| $O(2^n)$      | exponential  | squared                            |

_sorted from fastest to slowest_

If mult/ div by contant c: $log_c(n)$ e.g. `for (int i = 0; i < n; i *= 2)`
If add/ sub by constant c: $n/c$ e.g. `for (int i = 0; i < n; i += 2)`

- We write $O(f(n))$ for some function $f(n)$.
- You get the doubling time by taking $f(2n)/f(n)$.
- E.g. if $f(n)=n^3$, then $f(2n)/f(n)=(2n)^3/n^3=8$.
  - So if you double $n$, the running time goes up 8x.
- For $O(2^n)$, increasing $n$ by 1 causes the runtime to double!

Note: these are **common** cases of big O, but this list is not exhaustive.

<!-- #### Big O rules

- $O(f(n)+g(n))=\max(O(f(n)),O(g(n)))$
- $O(f(n)g(n))=O(f(n))O(g(n))$
- $O(c^nf(n))=O(f(n))$ for any constant $c>1$ -->

## Space Complexity

- Space complexity is the amount of memory used by an algorithm.
- We can use big O notation to describe space complexity.

### range vs list vs arange

- `range()` is a generator, so it doesn't take up memory
- `list(range())` is a list, so it takes up memory
- `np.arange()` is an array, so it takes up memory

## Searching

| Feature                       | Linear Search                                                              | Binary Search                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Principle**                 | Sequentially checks each element until a match is found or end is reached. | Repeatedly divides in half the portion of the list that could contain the item until you've narrowed down the possible locations to just one. |
| **Best-case Time Complexity** | \(O(1)\)                                                                   | \(O(1)\)                                                                                                                                      |
| **Space Complexity**          | \(O(1)\)                                                                   | \(O(1)\)                                                                                                                                      |
| **Works on**                  | Unsorted and sorted lists                                                  | Sorted lists only                                                                                                                             |

### Linear Search Code

```python
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1  # not found

# Example usage:
arr = [10, 20, 80, 30, 60, 50, 110, 100, 130, 170]
x = 110
result = linear_search(arr, x)
print("Element is present at index" if result != -1 else "Element is not present in array", result)

```

### Binary Search Code

```python
def binary_search(arr, l, r, x):
    while l <= r:
        mid = l + (r - l) // 2
        # Check if x is present at mid
        if arr[mid] == x:
            return mid
        # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1
        # If x is smaller, ignore right half
        else:
            r = mid - 1
    # Element was not present
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x = 70
result = binary_search(arr, 0, len(arr)-1, x)
print("Element is present at index" if result != -1 else "Element is not present in array", result)

```

## Sorting

| Algorithm          | Worst-case Time Complexity | Space Complexity | Description                                                                                                                                                                                               | Viz                                            |
| ------------------ | -------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Insertion Sort** | \(O(n^2)\)                 | \(O(1)\)         | Builds the final sorted list one item at a time. It takes one input element per iteration and finds its correct position in the sorted list.                                                              | <img src="img/insertion_sort.gif" width="200"> |
| **Selection Sort** | \(O(n^2)\)                 | \(O(1)\)         | Divides the input list into two parts: a sorted and an unsorted sublist. It repeatedly selects the smallest (or largest) element from the unsorted sublist and moves it to the end of the sorted sublist. | <img src="img/selection_sort.gif" width="200"> |
| **Bubble Sort**    | \(O(n^2)\)                 | \(O(1)\)         | Repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The process is repeated for each item.                                                       | <img src="img/bubble_sort.gif" width="200">    |
| **Merge Sort**     | \(O(n \log n)\)            | \(O(n)\)         | Divides the unsorted list into n sublists, each containing one element, then repeatedly merges sublists to produce new sorted sublists until there is only one sublist remaining.                         | <img src="img/merge_sort.gif" width="200">     |
| **Heap Sort**      | \(O(n \log n)\)            | \(O(1)\)         | Converts the input data into a heap data structure. It then extracts the topmost element (max or min) and reconstructs the heap, repeating this process until the heap is empty.                          | <img src="img/heap_sort.gif" width="200">      |

gifs from https://emre.me/algorithms/sorting-algorithms/

## HashMap

### Hashing

- Hashing is a technique that is used to uniquely identify a specific object from a group of similar objects.
- in python: `hash()`
- only immutable objects can be hashed
  - lists, sets, and dictionaries are mutable and cannot be hashed
  - tuples are immutable and can be hashed

### Python Dict

- Creating dictionaries:
  - `x = {}`
    - `x = {'a': 1, 'b': 2}`
  - `x = dict()`
    - `x = dict(a=1, b=2)`
    - `x = dict([('a', 1), ('b', 2)])`
    - `x = dict(zip(['a', 'b'], [1, 2]))`
    - `x = dict({'a': 1, 'b': 2})`
- Accessing values:
  - `x['a']`: if key is not found, raises `KeyError`
  - `x.get('a', 0)`: returns `0` if key is not found

### Python Defaultdict

- `defaultdict` is a subclass of `dict` that returns a default value when a key is not found
  - `from collections import defaultdict`
  - `d = defaultdict(int)`
    - `d['a']` returns `0`
  - `d = defaultdict(list)`
    - `d['a']` returns `[]`
    - not `list()` because `list()` is a function that returns an empty list, `list` is a type
  - `d = defaultdict(set)`
    - `d['a']` returns `set()`
  - `d = defaultdict(lambda: "hello I am your friendly neighbourhood default value")`
    - `d['a']` returns `"hello I am your friendly neighbourhood default value"`

### Python Counter

- `Counter` is a subclass of `dict` that counts the number of occurrences of an element in a list

  - `from collections import Counter`
  - `c = Counter(['a', 'b', 'c', 'a', 'b', 'b'])`
    - `c['a']` returns `2`
    - `c['b']` returns `3`
    - `c['c']` returns `1`
    - `c['d']` returns `0`
  - `c = Counter({'a': 2, 'b': 3, 'c': 1})`
  - `c = Counter(a=2, b=3, c=1)`

- other functions:
  - `c.most_common(2)` returns the 2 most common elements in the list: `[('b', 3), ('a', 2)]`

## Graphs

- contains vertices (or nodes) and edges
- use networkx to create graphs in Python

```python
import networkx as nx

G = nx.Graph()  # create empty graph
G.add_node("YVR")  # add node 1
G.add_node("YYZ")  # add node 2
G.add_node("YUL")  # add node 3

G.add_edge("YVR", "YYZ", weight=4)  # add edge between node 1 and node 2
G.add_edge("YVR", "YUL", weight=5)  # add edge between node 1 and node 3

nx.draw(G, with_labels=True) # draw graph but random layout
nx.draw(G, with_labels=True, pos=nx.spring_layout(G, seed=5)) # not random layout
```

### Other graphs

- directed graph: edges have direction
  - `nx.DiGraph()`

### Terminology

- Degree: number of edges connected to a node
- Path: sequence of nodes connected by edges
- Connected: graph where there is a path between every pair of nodes
- Component: connected subgraph

## Graph Searching

### Breadth-first search (BFS)

```python
def bfs(graph, start, search):
    # graph: networkx graph

    visited = set()
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == search:
            return True
        for neighbour in g.neighbors(node):
            queue.append(neighbour)

    return False
```

#### Queue

- FIFO (first in first out)

```python
queue = []
queue.append(1)  # add to end of queue
queue.append(2)  # add to end of queue
queue.pop(0)  # remove from front of queue
```

### Depth-first search (DFS)

#### Stack

- LIFO (last in first out)

```python
stack = []
stack.append(1)  # add to end of stack
stack.append(2)  # add to end of stack
stack.pop()  # remove from end of stack
```

### Graph representation

- Adjacency List
  - lists of all the edges in the graph
  - space complexity: O(E)
  - still need to store all the nodes
- Adjacency Matrix

  - matrix of 0s and 1s with size V x V
  - dense matrix space complexity: O(V^2)
  - sparse matrix space complexity: O(E)
    - Can do both directed and undirected graphs

- networkx uses scipy sparse matrix
  - create sparse matrix: `scipy.sparse.csr_matrix(x)`
  - need to acces using `matrix[row, col]`
    - `matrix[row][col]` will not work, in np it will work
  - to sum all the rows: `matrix.sum(axis=1)`
    - cannot do `np.sum(matrix, axis=1)`
    - or do `matrix.getnnz(axis=1)` to get number of non-zero elements in each row
      - to find vertex: `np.argmax(matrix.getnnz(axis=1))`
      - to find # max edges: `np.max(matrix.getnnz(axis=1))`

## Linear Programming

- **Linear programming** is a method to achieve the best outcome (e.g. maximum profit or lowest cost) in a mathematical model whose requirements are represented by linear relationships

## Defining the problem

To specify an optimization problem, we need to specify a few things:

1. A specification of the space of **possible inputs**.
2. An _**objective function**_ which takes an input and computes a score.
3. (Optional) A set of _constraints_, which take an input and return true/false.
   - can only get same or worse score than not having constraints
4. Are we **maximizing or minimizing**?

## Using Python PuLP

- PuLP is an LP modeler written in Python
- continuous problems are easier to solve than discrete problems
  - discrete problems might not be "optimal"
- Example of discrete problem: assigning TAs to courses

```python
# Define the problem
prob = pulp.LpProblem("TA-assignments", pulp.LpMaximize)

# Define the variables
x = pulp.LpVariable.dicts("x", (TAs, courses), 0, 1, pulp.LpInteger)

# add constraints
for course in courses:
    prob += pulp.lpSum(x[ta][course] for ta in TAs) == 1 # += adds constraint

# add objective
prob += pulp.lpSum(x[ta][course] * happiness[ta][course] for ta in TAs for course in courses)

# solve
prob.solve()
# prob.solve(pulp.apis.PULP_CBC_CMD(msg=0)) # to suppress output

# check status
pulp.LpStatus[prob.status] # 'Optimal'

# print results
for ta in TAs:
    for course in courses:
        if x[ta][course].value() == 1.0:
            print(f"{ta} is assigned to {course}")

```

## Dynamic Programming

- Dynamic programming is a method for solving a complex problem by breaking it down into a collection of simpler subproblems, solving each of those subproblems just once, and storing their solutions using a memory-based data structure (array, map, etc).
- Dynamic programming only works on problems with optimal substructure and overlapping subproblems.
  - optimal substructure: optimal solution can be constructed from optimal solutions of its subproblems
  - overlapping subproblems: subproblems recur many times
- Dynamic programming is usually applied to optimization problems.
