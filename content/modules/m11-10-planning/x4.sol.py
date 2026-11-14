"""Alıştırma8.4: yalnız seçim bonusunu, planlama bonusundan ayır."""
import math
import random

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def transition(s, a, kind, changed):
    r, c = divmod(s, 9)
    dr, dc = MOVES[a]
    y, x = r + dr, c + dc
    wall = y == 3 and ((x >= 1 if changed else x <= 7) if kind == "blocking"
                       else x >= 1 and (x <= 7 if changed else True))
    z = s
    if 0 <= y < 6 and 0 <= x < 9 and not wall:
        z = 9*y + x
    return z, float(z == 8), z == 8


def select_action(qs, ages, kappa, rng):
    scores = [q + kappa*math.sqrt(t) for q, t in zip(qs, ages)]
    best = max(scores)
    return rng.choice([a for a, score in enumerate(scores) if score == best])


def run_bonus(kind="shortcut", variant="selection", n_steps=6000, change=3000,
              kappa=.001, seed=0, n_plan=50, alpha=1., epsilon=.1):
    """selection: ε yok, updates bonussuz. plus: bonus yalnız model yedeğinde."""
    if variant not in {"selection", "plus", "plain"} or kind not in {"shortcut", "blocking"}:
        raise ValueError("variant/kind")
    if kappa < 0 or n_plan < 0 or not 0 < alpha <= 1:
        raise ValueError("κ≥0, n≥0, 0<α≤1")
    behaviour, planning = random.Random(seed), random.Random(seed+100000)
    Q = [[0.]*4 for _ in range(54)]
    model, actions, last = {}, {}, {}
    start, s, total = 48, 48, 0.
    cumulative = []

    def put(s, a, outcome):
        if s not in actions:
            actions[s] = []
        if (s, a) not in model:
            actions[s].append(a)
        model[s, a] = outcome

    def update(s, a, reward, z, done):
        Q[s][a] += alpha*(reward + (0. if done else .95*max(Q[z]))-Q[s][a])

    for time in range(1, n_steps+1):
        ages = [time-last.get((s, a), 0) for a in range(4)]
        if variant == "selection":
            a = select_action(Q[s], ages, kappa, behaviour)
        elif behaviour.random() < epsilon:
            a = behaviour.randrange(4)
        else:
            a = select_action(Q[s], ages, 0, behaviour)
        z, reward, done = transition(s, a, kind, time > change)
        update(s, a, reward, z, done)
        if variant == "plus" and s not in actions:
            for b in range(4):
                put(s, b, (s, 0., False))
        put(s, a, (z, reward, done))
        last[s, a] = time  # Sadece gerçek deneyim; planlama bu saati değiştirmez.
        for _ in range(n_plan):
            ps = planning.choice(list(actions))
            pa = planning.choice(actions[ps])
            pz, pr, pd = model[ps, pa]
            if variant == "plus":
                pr += kappa*math.sqrt(time-last.get((ps, pa), 0))
            update(ps, pa, pr, pz, pd)
        total += reward  # Bonussuz gerçek çevre ödülü.
        cumulative.append(total)
        s = start if done else z
    return {"cumulative": cumulative, "Q": Q, "last": last, "model": model}


print(run_bonus(n_steps=600, change=300, n_plan=5)["cumulative"][-1])
