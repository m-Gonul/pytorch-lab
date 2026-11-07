import random


def simulate(policy, p_r, p_w, ticks, seed=0):
    rng = random.Random(seed)
    readers_in, writer_left = [], 0
    qr, qw = [], []
    wr, ww, done_w = [], [], 0
    for t in range(ticks):
        if rng.random() < p_r:
            qr.append(t)
        if rng.random() < p_w:
            qw.append(t)
        readers_in = [x - 1 for x in readers_in if x > 1]
        if writer_left:
            writer_left -= 1
            if writer_left == 0:
                done_w += 1
        if writer_left == 0:
            if not readers_in and qw and (policy == "writer" or not qr):
                ww.append(t - qw.pop(0))
                writer_left = 5
            elif qr and not (policy == "writer" and qw):
                while qr:
                    wr.append(t - qr.pop(0))
                    readers_in.append(3)
    avg = lambda v: sum(v) / len(v) if v else float("inf")
    return avg(ww), avg(wr), done_w


print(simulate("reader", 0.6, 0.05, 5000))
