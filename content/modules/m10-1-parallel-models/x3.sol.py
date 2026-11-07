import heapq
import math
import random


def simulate(lam, mu, c, n_customers, seed=0):
    rng = random.Random(seed)
    t, events = 0.0, []
    for i in range(n_customers):
        t += rng.expovariate(lam)
        heapq.heappush(events, (t, 0, i))
    free_servers, queue, waits, svc = c, [], [], {}
    while events:
        now, kind, i = heapq.heappop(events)
        if kind == 0:
            if free_servers:
                free_servers -= 1
                waits.append(0.0)
                heapq.heappush(events, (now + rng.expovariate(mu), 1, i))
            else:
                queue.append((now, i))
        else:
            if queue:
                arr, j = queue.pop(0)
                waits.append(now - arr)
                heapq.heappush(events, (now + rng.expovariate(mu), 1, j))
            else:
                free_servers += 1
    return sum(waits) / len(waits)


def erlang_c_wait(lam, mu, c):
    a = lam / mu
    rho = a / c
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    C = top / (s + top)
    return C / (c * mu - lam)


print(simulate(1.8, 1.0, 2, 20000), erlang_c_wait(1.8, 1.0, 2))
