import heapq
import math


def huffman(counts):
    V = len(counts)
    heap = [(c, i, ("leaf", i)) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    nxt = 0
    tick = V
    children = {}
    while len(heap) > 1:
        c1, _, a = heapq.heappop(heap)
        c2, _, b = heapq.heappop(heap)
        node = ("inner", nxt)
        children[nxt] = (a, b)
        nxt += 1
        heapq.heappush(heap, (c1 + c2, tick, node))
        tick += 1
    root = heap[0][2]
    codes, paths = {}, {}
    stack = [(root, [], [])]
    while stack:
        node, code, path = stack.pop()
        if node[0] == "leaf":
            codes[node[1]], paths[node[1]] = code, path
        else:
            a, b = children[node[1]]
            stack.append((a, code + [0], path + [node[1]]))
            stack.append((b, code + [1], path + [node[1]]))
    return codes, paths


def hs_prob(v, w, codes, paths, Wn):
    p = 1.0
    for bit, n in zip(codes[w], paths[w]):
        s = sum(a * b for a, b in zip(Wn[n], v))
        p *= 1 / (1 + math.exp(-s if bit == 0 else s))
    return p
