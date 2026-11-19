import math


class PortableRNG:
    """Verilen Mulberry32 aracı: C++/Python/JS deney akışını eşler."""
    def __init__(self, seed):
        self.state = seed

    def random(self):
        self.state = (self.state + 0x6d2b79f5) & 0xffffffff
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296


def baird_q(steps=100000, alpha=.01, seed=1):
    rng = PortableRNG(seed)
    Q = [[1.] * 6 + [10., 1.] for _ in range(2)]
    visits = [[0, 0] for _ in range(7)]
    state = int(rng.random() * 7)
    points, norms = [0], [math.sqrt(sum(v*v for row in Q for v in row))]

    def value(s, a):
        return 2*Q[a][s]+Q[a][7] if s < 6 else Q[a][6]+2*Q[a][7]

    stride = max(1, steps//200)
    for t in range(steps):
        action = int(rng.random() >= 6/7)
        new_state = 6 if action else int(rng.random()*6)
        delta = .99*max(value(new_state, 0), value(new_state, 1))-value(state, action)
        index, first, last = (state, 2, 1) if state < 6 else (6, 1, 2)
        Q[action][index] += alpha*delta*first
        Q[action][7] += alpha*delta*last
        visits[state][action] += 1
        state = new_state
        if (t+1) % stride == 0 or t+1 == steps:
            points.append(t+1)
            norms.append(math.sqrt(sum(v*v for row in Q for v in row)))
    return dict(Q=Q, visits=visits, points=points, norms=norms, actual_steps=steps)


print(baird_q(10000)["norms"][-1])
