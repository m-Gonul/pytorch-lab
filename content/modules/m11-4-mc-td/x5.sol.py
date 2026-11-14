import math
import random


def episodes19(count, seed=0):
    rng = random.Random(seed)
    episodes = []
    for _ in range(count):
        state, path = 10, [10]
        while state not in (0, 20):
            state += 1 if rng.random() < 0.5 else -1
            path.append(state)
        episodes.append(path)
    return episodes


def compare_nstep(episodes, n, alpha):
    result = {}
    for stored_delta, name in [(False, "target"), (True, "delta")]:
        V = [0.0] * 21
        history = []
        for path in episodes:
            T = len(path) - 1
            reward = 1.0 if path[-1] == 20 else -1.0
            prefix = [0.0]
            for t in range(T + n - 1):
                if stored_delta and t < T:
                    delta = (reward if t == T - 1 else V[path[t + 1]]) - V[path[t]]
                    prefix.append(prefix[-1] + delta)
                tau = t - n + 1
                if tau < 0:
                    continue
                h = min(tau + n, T)
                if stored_delta:
                    error = prefix[h] - prefix[tau]
                else:
                    G = reward if h == T else V[path[h]]
                    error = G - V[path[tau]]
                V[path[tau]] += alpha * error
            history.append(math.sqrt(sum((V[i] - (i / 10 - 1)) ** 2 for i in range(1, 20)) / 19))
        result[name] = {"V": V, "rms": history}
    return result


if __name__ == "__main__":
    out = compare_nstep(episodes19(10), 4, 0.4)
    print({name: sum(v["rms"]) / len(v["rms"]) for name, v in out.items()})
