import random


def cas(mem, key, expected, new):
    if mem[key] == expected:
        mem[key] = new
        return True
    return False


def push(mem, node, tagged):
    while True:
        old = mem["top"]
        yield
        mem["next"][node] = old[0]
        yield
        if cas(mem, "top", old, (node, old[1] + 1 if tagged else 0)):
            return
        mem["retries"] += 1
        yield


def pop(mem, tagged):
    while True:
        old = mem["top"]
        yield
        if old[0] is None:
            return None
        nxt = mem["next"][old[0]]
        yield
        if cas(mem, "top", old, (nxt, old[1] + 1 if tagged else 0)):
            return old[0]
        mem["retries"] += 1
        yield


def worker(mem, rounds, tagged):
    for _ in range(rounds):
        node = yield from pop(mem, tagged)
        if node is not None:
            yield
            yield from push(mem, node, tagged)


def schedule(gens, seed, max_steps=100000):
    rng = random.Random(seed)
    live = list(gens)
    steps = 0
    while live and steps < max_steps:
        g = rng.choice(live)
        try:
            next(g)
        except StopIteration:
            live.remove(g)
        steps += 1
    return steps


def stack_ok(mem, nodes):
    seen, p = [], mem["top"][0]
    while p is not None and len(seen) <= len(nodes):
        seen.append(p)
        p = mem["next"][p]
    return p is None and sorted(seen) == sorted(nodes)
