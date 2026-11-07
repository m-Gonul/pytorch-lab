import math
import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    text = ("ali ata bak. ayse ip atla. ali topu at. ayse kitap oku. ali ile ayse parka gitti. "
            "kedi agaca cikti. kopek havladi. kus ucup gitti. ") * 12
    chars = sorted(set(text)); V = len(chars)
    ids = torch.tensor([chars.index(c) for c in text])

    def cn():
        N = torch.zeros(V, V)
        for a, b in zip(ids[:-1].tolist(), ids[1:].tolist()):
            N[a, b] += 1
        ref = -sum(math.log(N[a, b] / N[a].sum()) for a, b in zip(ids[:-1].tolist(), ids[1:].tolist())) / (len(ids) - 1)
        close(need(ns, "count_nll")(ids, V), ref, rtol=1e-5)
    R.test("count_nll = sayım modelinin NLL'i", cn)

    def tr():
        torch.manual_seed(0)
        m, loss = need(ns, "train_bigram")(ids, V, steps=500, lr=0.5)
        ensure(isinstance(m, nn.Module) and any(isinstance(x, nn.Embedding) for x in m.modules()), "nn.Embedding içeren bir model bekleniyordu")
        opt = need(ns, "count_nll")(ids, V)
        ensure(abs(loss - opt) < 0.02, f"sinir ağı kaybı {loss:.4f}, sayım optimumu {opt:.4f}")
        return f"sinir ağı {loss:.4f} ≈ sayım MLE {opt:.4f} (aynı model, iki yol)"
    R.test("bigram ağı sayım MLE'sine yakınsıyor", tr)

    def sparse():
        torch.manual_seed(0)
        m = need(ns, "BigramLM", "sınıf")(V)
        x = torch.tensor([3, 3, 5]); y = torch.tensor([1, 2, 0])
        torch.nn.functional.cross_entropy(m(x), y).backward()
        W = [p for p in m.parameters()][0]
        rows = (W.grad.abs().sum(1) > 0).nonzero().view(-1).tolist()
        ensure(rows == [3, 5], f"yalnız kullanılan satırlar gradyan almalı: {rows}")
    R.test("embedding gradyanı yalnız kullanılan satırlarda", sparse)
