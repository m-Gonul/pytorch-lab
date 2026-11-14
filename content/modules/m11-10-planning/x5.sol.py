"""Alıştırma8.8: yörünge dağılımını, yedeğin örneklenmesinden ayır."""
import math
import numpy as np


class U32:
    def __init__(self, seed):
        self.state = seed or 1

    def random(self):
        self.state = (self.state+0x6d2b79f5) & 0xffffffff
        t = ((self.state ^ (self.state >> 15))*(1 | self.state)) & 0xffffffff
        t = (((t+((t ^ (t >> 7))*(61 | t) & 0xffffffff)) & 0xffffffff) ^ t) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff)/4294967296

    def index(self, n):
        return int(self.random()*n)

    def normal(self):
        u, v = self.random(), self.random()
        return math.sqrt(-2*math.log(max(u, 1e-15)))*math.cos(2*math.pi*v)


def make_task(S, b, seed=800001):
    if not 1 <= b <= S:
        raise ValueError("1≤b≤S")
    rng = U32(seed)
    next_states = np.empty((S, 2, b), dtype=np.int64)
    rewards = np.empty((S, 2, b))
    for s in range(S):
        for a in range(2):
            selected = set()
            for j in range(b):
                z = rng.index(S)
                while z in selected:
                    z = rng.index(S)
                selected.add(z)
                next_states[s, a, j], rewards[s, a, j] = z, rng.normal()
    return next_states, rewards


def expected_backup(Q, next_states, rewards, s, a):
    return .9*np.mean(rewards[s, a]+np.max(Q[next_states[s, a]], axis=1))


def evaluate_policy(Q, next_states, rewards, tol=1e-11):
    S = len(Q)
    actions = np.argmax(Q, axis=1)  # Eşitlikte ilk eylem; davranıştan ayrı tanım.
    successors = next_states[np.arange(S), actions]
    r = .9*np.mean(rewards[np.arange(S), actions], axis=1)
    V = np.zeros(S)
    for _ in range(10000):
        new = r+.9*np.mean(V[successors], axis=1)
        difference = np.max(np.abs(new-V))
        V = new
        if difference <= tol*.1:
            break
    residual = np.max(np.abs(V-r-.9*np.mean(V[successors], axis=1)))
    if residual > tol:
        raise ArithmeticError("politika değerlendirmesi yakınsamadı")
    return V, float(residual)


def learn(next_states, rewards, steps, mode="trajectory", seed=1700001, points=None):
    if mode not in {"uniform", "trajectory"}:
        raise ValueError("mode")
    S, _, b = next_states.shape
    Q, rng, state = np.zeros((S, 2)), U32(seed), 0
    points = [0, steps] if points is None else sorted(set(points))
    values, residuals, index = [], [], 0
    for t in range(steps+1):
        if index < len(points) and t == points[index]:
            V, residual = evaluate_policy(Q, next_states, rewards)
            values.append(float(V[0]))
            residuals.append(residual)
            index += 1
        if t == steps:
            break
        if mode == "uniform":
            s, a = divmod(t % (2*S), 2)
            Q[s, a] = expected_backup(Q, next_states, rewards, s, a)
        else:
            if rng.random() < .1:
                a = rng.index(2)
            elif Q[state, 0] == Q[state, 1]:
                a = rng.index(2)
            else:
                a = int(Q[state, 1] > Q[state, 0])
            # İki yöntemde de aynı tam BEKLENEN hedef. Yalnız ziyaret dağılımı farklı.
            Q[state, a] = expected_backup(Q, next_states, rewards, state, a)
            state = 0 if rng.random() < .1 else next_states[state, a, rng.index(b)]
    return {"Q": Q, "values": values, "residuals": residuals, "backups": steps}


def experiment(S=10000, b=1, runs=200, steps=200000, seed=800001, points=None):
    points = [0, steps//20, steps//5, steps] if points is None else points
    measurements = np.empty((runs, 2, len(points)))
    maximum_residual = 0.
    for i in range(runs):
        next_states, rewards = make_task(S, b, seed+i)
        for j, mode in enumerate(["uniform", "trajectory"]):
            result = learn(next_states, rewards, steps, mode, seed+i+900000, points)
            measurements[i, j] = result["values"]
            maximum_residual = max(maximum_residual, max(result["residuals"]))
    se = measurements.std(axis=0, ddof=1)/math.sqrt(runs) if runs > 1 else np.zeros_like(measurements[0])
    return {"points": points, "mean": measurements.mean(axis=0), "se": se,
            "maximum_residual": maximum_residual, "backups": 2*runs*steps}


# Notebook/otomatik kontrol küçük örneği çalıştırır. Tam kaynak bütçesini yukarıdaki
# fonksiyonla çağırın; depodaki C++ hızlandırıcısı bu deneyin 200 görevini gerçekten koştu.
print(experiment(S=17, b=3, runs=1, steps=100)["mean"])
