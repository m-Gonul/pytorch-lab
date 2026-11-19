import math


class PortableRNG:
    """Mulberry32: bu ekin C++/Python/JS yollarını eşlemek için verilen araç."""
    def __init__(self, seed):
        self.state = seed

    def random(self):
        self.state = (self.state + 0x6d2b79f5) & 0xffffffff
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296


def source_tiles(position, velocity, action):
    qx = math.floor(8 * (8 * position / 1.7))
    qv = math.floor(8 * (8 * velocity / .14))
    return [((t*10 + (qx+t)//8 + 6)*11 + (qv+3*t)//8 + 4)*3 + action
            for t in range(8)]


def mountaincar_source(episodes=20, n=1, alpha=.5, seed=1010001, budget=2000000):
    if n < 1 or budget < 1:
        raise ValueError("n ve geçiş bütçesi pozitif olmalı")
    starts, actions = PortableRNG(seed), PortableRNG(seed+99173)
    w, lengths, physical = [0.] * 2640, [], 0

    def features(x, v, a):
        return source_tiles(x, v, a)

    def q(F):
        return sum(w[i] for i in F)

    def choose(x, v):
        values = [q(features(x, v, a)) for a in range(3)]
        ties = [a for a in range(3) if values[a] == max(values)]
        return ties[int(actions.random()*len(ties))]

    for _ in range(episodes):
        x, v = -.6+.2*starts.random(), 0.
        states, acts = [None]*(n+1), [0]*(n+1)
        states[0], acts[0] = (x, v), choose(x, v)
        T, t = math.inf, 0
        while True:
            cutoff = False
            if t < T:
                a = acts[t % (n+1)]
                v = min(.07, max(-.07, v+.001*(a-1)-.0025*math.cos(3*x)))
                x = min(.5, max(-1.2, x+v))
                if x <= -1.2:
                    v = 0.
                physical += 1
                states[(t+1) % (n+1)] = (x, v)
                if x >= .5:
                    T = t+1
                else:
                    acts[(t+1) % (n+1)] = choose(x, v)
                    cutoff = physical >= budget
            tau = t-n+1
            if tau >= 0:
                end = min(T, tau+n)
                G = -(end-tau)
                if tau+n < T:
                    bs = states[(tau+n) % (n+1)]
                    G += q(features(*bs, acts[(tau+n) % (n+1)]))
                F = features(*states[tau % (n+1)], acts[tau % (n+1)])
                delta = alpha/8 * (G-q(F))
                for i in F:
                    w[i] += delta
            if cutoff:
                return dict(w=w, lengths=lengths, actual_steps=physical, cutoff=True,
                            unfinished_steps=t+1, last_state=(x, v))
            if tau == T-1:
                lengths.append(int(T))
                break
            t += 1
        if physical >= budget and len(lengths) < episodes:
            return dict(w=w, lengths=lengths, actual_steps=physical, cutoff=True,
                        unfinished_steps=0, last_state=(x, v))
    return dict(w=w, lengths=lengths, actual_steps=physical, cutoff=False,
                unfinished_steps=0, last_state=(x, v))


print(mountaincar_source(5)["lengths"])
