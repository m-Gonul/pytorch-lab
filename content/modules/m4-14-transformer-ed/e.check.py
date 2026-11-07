import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labcheck import need, ensure, fmt_time


class _TF(nn.Module):
    def __init__(self, V=50, d=128, N=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.core = nn.Transformer(d, 4, N, N, 4 * d, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d, V)

    def encode(self, x):
        return self.core.encoder(self.emb(x)), None

    def decode(self, y, mem, pad):
        T = y.shape[1]
        return self.out(self.core.decoder(self.emb(y), mem, tgt_mask=torch.triu(torch.full((T, T), float("-inf")), 1)))

    def forward(self, x, y):
        mem, pad = self.encode(x)
        return self.decode(y, mem, pad)


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        m = _TF().eval()
        rows = []
        for S, T in ((16, 8), (128, 16), (512, 32)):
            x = torch.randint(2, 50, (16, S))
            t0 = time.perf_counter()
            a = need(ns, "greedy_naive")(m, x, T)
            ta = time.perf_counter() - t0
            t0 = time.perf_counter()
            b = need(ns, "greedy_cached")(m, x, T)
            tb = time.perf_counter() - t0
            ensure(torch.equal(a, b) and tuple(a.shape) == (16, T), "aynı çıktı")
            rows.append([S, T, fmt_time(ta), fmt_time(tb), f"{ta / tb:.1f}×"])
        R_.table("Açgözlü çözümleme, batch 16, d = 128, 4 + 4 katman (CPU)", ["kaynak uzunluğu S", "üretilen T", "kodlayıcı her adımda", "bellek bir kez", "hızlanma"], rows,
                 note="Kodlayıcı çıktısı çözümleme boyunca değişmez: bir kez hesaplanıp N kod çözücü katmanına verilir. Yeniden hesaplamak T × kodlayıcı maliyeti (S² attention) ekler; kaynak uzadıkça fark büyür. Ek olarak cross-attention'ın K, V izdüşümleri de önbelleğe alınabilir, öz-attention için KV-cache (M4.7) kod çözücü tarafını hızlandırır.")
    R_.test("kodlayıcı belleğini yeniden kullanma", t)
