import random


def differential_q_demo(steps=50000, seed=0, eta=0.1):
    rng = random.Random(seed)
    Q = [[0., 0.], [0., 0.]]
    counts = [[0, 0], [0, 0]]
    state, rho, total = 0, 0., 0.
    for _ in range(steps):
        action = rng.randrange(2)  # davranış politikası; bütün çiftleri kapsar
        if state == 0:
            reward, next_state = (2., 0) if action == 0 else (0., 1)
        else:
            reward, next_state = 4., int(rng.random() >= .5)
        alpha = .2 / (1 + counts[state][action])**.6
        delta = reward-rho+max(Q[next_state])-Q[state][action]
        Q[state][action] += alpha*delta
        rho += eta*alpha*delta
        counts[state][action] += 1
        total += reward
        state = next_state
    return dict(Q=Q, rho=rho, visits=counts, behavior_reward=total/steps)


print(differential_q_demo()["rho"])
