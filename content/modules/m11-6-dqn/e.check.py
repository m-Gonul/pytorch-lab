import math
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time

G_, MC, MP, L, FM, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02


def loop_len(w, x, max_len=500):
    for t in range(max_len):
        a = 1 if sum(xi * wi for xi, wi in zip(x, w)) > 0 else 0
        pos, vel, th, om = x
        f = FM if a else -FM
        ct, st = math.cos(th), math.sin(th)
        temp = (f + MP * L * om * om * st) / (MC + MP)
        thacc = (G_ * st - ct * temp) / (L * (4 / 3 - MP * ct * ct / (MC + MP)))
        xacc = temp - MP * L * thacc * ct / (MC + MP)
        x = [pos + TAU * vel, vel + TAU * xacc, th + TAU * om, om + TAU * thacc]
        if abs(x[0]) > 2.4 or abs(x[2]) > 12 * 2 * math.pi / 360:
            return t + 1
    return max_len


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: max_loops(src, 1, func="rollout_lengths"))
    g = torch.Generator().manual_seed(4)
    W = torch.rand(40, 4, generator=g, dtype=torch.float64) * 2 - 1
    x0 = (torch.rand(40, 5, 4, generator=g, dtype=torch.float64) - 0.5) * 0.1

    def eq():
        Lv = need(ns, "rollout_lengths")(W, x0)
        ref = torch.tensor([sum(loop_len(W[p].tolist(), x0[p, e].tolist()) for e in range(5)) / 5 for p in range(40)], dtype=torch.float64)
        close(Lv.numpy(), ref.numpy(), atol=1e-9, what="döngülü değerlendirmeyle aynı uzunluklar")
    R_.test("rollout_lengths", eq)

    def rs():
        Wb, Lb = need(ns, "random_search")(4000, 10, 0)
        ensure(Wb.shape == (4000, 4) and Lb.shape == (4000,), "şekiller")
        ensure(float(Lb.max()) >= 499 and float(Lb.median()) < 100, "en iyi politika 500'e ulaşmalı; çoğu kötü")
        best = Wb[Lb.argmax()]
        x0b = (torch.rand(1, 100, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64) - 0.5) * 0.1
        gen = float(ns["rollout_lengths"](best[None], x0b)[0])
        t_vec = bench(lambda: ns["random_search"](4000, 10, 0), repeat=1, number=1)
        t_loop = bench(lambda: [loop_len(Wb[p].tolist(), [0.01, 0, 0.01, 0]) for p in range(200)], repeat=1, number=1) * 20 * 10
        R_.table("4000 doğrusal politika × 10 ortam", ["ölçü", "değer"],
                 [["500'e ulaşan politika", int((Lb >= 499).sum())], ["ortanca uzunluk", f"{float(Lb.median()):.0f}"],
                  ["en iyi politikanın 100 yeni ortamda ort.", f"{gen:.0f}"], ["süre: tensör", fmt_time(t_vec)], ["süre: politika döngüsü (tahmini)", fmt_time(t_loop)]],
                 note="CartPole doğrusal bir politikayla çözülebilir; 4 parametreli bir 'ağ' için rastgele arama DQN'den çok daha ucuzdur (Mania ve ark. 2018). Hangi yöntemin seçileceği problemin yapısına bağlıdır.")
    R_.test("random_search", rs)
