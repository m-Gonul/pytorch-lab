import torch
from labcheck import close, need, ensure


def check(ns, R):
    GPT, make_dataset = need(ns, "GPT", "sınıf"), need(ns, "make_dataset")

    def lr():
        f = need(ns, "lr_at")
        close([f(49, 100, 1100, 1e-3, 1e-4), f(600, 100, 1100, 1e-3, 1e-4), f(1100, 100, 1100, 1e-3, 1e-4)], [5e-4, 5.5e-4, 1e-4])
    R.test("lr_at", lr)

    def slv():
        torch.manual_seed(0)
        m = GPT().eval()
        D = make_dataset(7, 1)
        out = need(ns, "solve")(m, D[:, :6])
        ensure(tuple(out.shape) == (7, 3), "çıktı (B, 3)")
        idx = D[:, :6]
        with torch.no_grad():
            for _ in range(3):
                idx = torch.cat([idx, m(idx)[:, -1].argmax(-1, keepdim=True)], 1)
        ensure(torch.equal(out, idx[:, -3:]), "greedy çözme")
    R.test("solve (greedy, batch'li)", slv)

    def tr():
        torch.manual_seed(0)
        m = GPT()
        loss = need(ns, "train")(m, make_dataset(20000, 0), steps=1500)
        acc = need(ns, "accuracy")(m, make_dataset(2000, 99))
        ensure(acc >= 0.95, f"test doğruluğu {acc:.3f}")
        n = sum(p.numel() for p in m.parameters())
        return f"{n:,} parametre, son kayıp {loss:.4f}, görülmemiş 2000 toplamada doğruluk {acc:.3f}"
    R.test("eğitim: iki basamaklı toplama", tr)

    def unseen():
        D = make_dataset(20000, 0)
        a = D[:, 0] * 10 + D[:, 1]; b = D[:, 3] * 10 + D[:, 4]
        seen = set((a * 100 + b).tolist())
        return f"eğitim kümesi 10,000 olası çiftin {len(seen):,} tanesini içeriyor; test çiftleri büyük ölçüde örtüşüyor, genelleme için hold-out çiftlerle de dene"
    R.test("not: veri örtüşmesi", unseen)
