import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def nr():
        r = torch.tensor([[1.0, 1], [1, 1], [1, 1]])
        d = torch.tensor([[0.0, 0], [0, 1], [0, 0]])
        out = need(ns, "nstep_returns")(r, d, torch.tensor([10.0, 10.0]), 0.5)
        close(out.numpy(), torch.tensor([[1 + 0.5 + 0.25 + 0.125 * 10, 1 + 0.5], [1 + 0.5 + 0.25 * 10, 1.0], [1 + 0.5 * 10, 1 + 0.5 * 10]]).numpy(), what="n adımlı getiriler (bitişte önyükleme yok)")
    R_.test("nstep_returns", nr)

    def ac():
        net = need(ns, "ActorCritic")()
        lo, v = net(torch.zeros(5, 4))
        ensure(tuple(lo.shape) == (5, 2) and tuple(v.shape) == (5,), "çıktı şekilleri")
    R_.test("ActorCritic", ac)

    def tr():
        net, lens = need(ns, "train_a2c")(1500, 32, 8, 0.99, 2e-3, 0.01, 0)
        ev = ns["evaluate"](net)
        first = sum(lens[:50]) / 50
        ensure(float(ev.mean()) >= 200, f"A2C CartPole'u öğrenmeli (açgözlü ort. {float(ev.mean()):.0f})")
        R_.table("A2C, 32 ortam × 8 adım × 1500 güncelleme (384 000 adım)", ["ölçü", "değer"],
                 [["ilk 50 bölüm ort. uzunluk", f"{first:.0f}"], ["son 50 bölüm (rastgele politika) ort.", f"{sum(lens[-50:]) / 50:.0f}"],
                  ["açgözlü değerlendirme (20 bölüm, en fazla 500)", f"{float(ev.mean()):.0f}"]],
                 note="Eğitimdeki uzunluklar örneklenen (rastgele) politikadan; açgözlü politika genelde daha iyidir. DQN'den farklı olarak tampon yok: politika üstü, her veri bir kez kullanılır; paralel ortamlar korelasyonu kırar.")
    R_.test("train_a2c", tr)
