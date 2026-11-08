import random
from collections import deque


class Sem:
    def __init__(self, n):
        self.n = n
        self.q = deque()


def sem_P(s, tid):
    if s.n > 0:
        s.n -= 1
        return True
    s.q.append(tid)
    return False


def sem_V(s):
    if s.q:
        return s.q.popleft()
    s.n += 1
    return None


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
    for x in items:
        yield ("P", empty)
        yield ("P", mutex)
        yield ("do", lambda x=x: buf.append(x))
        yield ("V", mutex)
        yield ("V", full)
    yield ("V", done)


def consumer(buf, empty, full, mutex, out):
    while True:
        yield ("P", full)
        yield ("P", mutex)
        box = []
        yield ("do", lambda: box.append(buf.popleft()))
        yield ("V", mutex)
        yield ("V", empty)
        if box[0] is None:
            return
        out.append(box[0])


def closer(buf, empty, full, mutex, done, n_prod, n_cons):
    for _ in range(n_prod):
        yield ("P", done)
    for _ in range(n_cons):
        yield ("P", empty)
        yield ("P", mutex)
        yield ("do", lambda: buf.append(None))
        yield ("V", mutex)
        yield ("V", full)
