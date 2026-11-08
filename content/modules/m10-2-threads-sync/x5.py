import random


def cas(mem, key, expected, new):
    ...


def push(mem, node, tagged):
    ...


def pop(mem, tagged):
    ...


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
