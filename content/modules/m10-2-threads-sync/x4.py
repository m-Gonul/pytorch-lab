import random
from collections import deque


class Sem:
    def __init__(self, n):
        self.n = n
        self.q = deque()


def sem_P(s, tid):
    ...


def sem_V(s):
    ...


def run(threads, seed, max_steps=20000):
    rng = random.Random(seed)
    req, blocked, granted = {}, set(), set()
    for i, t in enumerate(threads):
        try:
            req[i] = next(t)
        except StopIteration:
            pass
    steps = 0
    while req:
        runnable = [i for i in req if i not in blocked]
        if not runnable:
            return "deadlock"
        if steps >= max_steps:
            return "timeout"
        i = rng.choice(runnable)
        steps += 1
        op, arg = req[i]
        if op == "P":
            if i in granted:
                granted.discard(i)
            elif not sem_P(arg, i):
                blocked.add(i)
                continue
        elif op == "V":
            w = sem_V(arg)
            if w is not None:
                blocked.discard(w)
                granted.add(w)
        elif op == "do":
            arg()
        try:
            req[i] = threads[i].send(None)
        except StopIteration:
            del req[i]
    return "ok"


def producer(buf, empty, full, mutex, items, done):
    ...


def consumer(buf, empty, full, mutex, out):
    ...


def closer(buf, empty, full, mutex, done, n_prod, n_cons):
    ...
