import torch
from labcheck import close, need, ensure


def check(ns, R):
    Chain = need(ns, "Chain", "sınıf")
    R.test("discounted_returns", lambda: close(need(ns, "discounted_returns")([0.0, 0.0, 1.0], 0.5), [0.25, 0.5, 1.0]))

    def ep():
        torch.manual_seed(0)
        pi = need(ns, "Policy", "sınıf")()
        logps, rewards = need(ns, "run_episode")(pi, Chain())
        ensure(logps.dim() == 1 and len(logps) == len(rewards) and logps.requires_grad, "logps (T,) ve gradyanlı olmalı")
        ensure(all(r == 0 for r in rewards[:-1]), "ödül yalnız son adımda olabilir")
    R.test("run_episode", ep)

    def train():
        wins = 0
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            pi = need(ns, "Policy", "sınıf")()
            need(ns, "train_reinforce")(pi, Chain())
            env = Chain(); obs = env.reset(); steps = 0
            with torch.no_grad():
                while True:
                    obs, r, done = env.step(pi(obs).argmax(-1)); steps += 1
                    if done.item():
                        break
            wins += int(env.s.item() == 7 and steps == 4)
        ensure(wins >= 2, f"3 tohumdan {wins} tanesinde açgözlü politika 4 adımda hedefe gitti")
        return f"3 tohumdan {wins}'inde optimal politika (4 adımda sağ uç)"
    R.test("REINFORCE zinciri çözüyor", train)
