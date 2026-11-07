import torch
import torch.nn as nn
from labcheck import close, need, ensure


def check(ns, R_):
    def rep():
        B = need(ns, "Replay")(5, 2)
        B.push(torch.ones(3, 2), torch.tensor([0, 1, 0]), torch.tensor([1.0, 2, 3]), torch.zeros(3, 2), torch.tensor([False, True, False]))
        ensure(len(B) == 3, "uzunluk 3")
        B.push(torch.full((4, 2), 7.0), torch.tensor([1, 1, 1, 1]), torch.tensor([4.0, 5, 6, 7]), torch.ones(4, 2), torch.tensor([True] * 4))
        ensure(len(B) == 5, "kapasitede durmalı")
        s, a, r, s2, d = B.sample(1000, torch.Generator().manual_seed(0))
        ensure(s.shape == (1000, 2) and a.dtype == torch.long and d.dtype.is_floating_point, "şekil ve tipler")
        ensure(set(r.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}, "en eski iki geçiş üzerine yazılmalı")
        ensure(bool(((r == 3) == (s[:, 0] == 1)).all()) and bool(((r > 3) == (d == 1)).all()), "alanlar hizalı")
    R_.test("Replay", rep)

    def tgt():
        qo, qt = nn.Linear(2, 3, bias=False), nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            qo.weight.copy_(torch.tensor([[1.0, 0], [0, 1], [0.5, 0.5]]))
            qt.weight.copy_(torch.tensor([[2.0, 0], [0, 0.5], [1, 1]]))
        s2 = torch.tensor([[1.0, 3.0], [4.0, 0.0]])
        r = torch.tensor([1.0, 0.0]); d = torch.tensor([0.0, 1.0])
        y = need(ns, "td_target")(qt, qo, r, s2, d, 0.9, False)
        close(y.numpy(), torch.tensor([1 + 0.9 * 4.0, 0.0]).numpy(), what="DQN hedefi")
        y2 = ns["td_target"](qt, qo, r, s2, d, 0.9, True)
        close(y2.numpy(), torch.tensor([1 + 0.9 * 1.5, 0.0]).numpy(), what="çift DQN hedefi (çevrimiçi ağ eylem 1'i seçer, hedef ağ 1.5 der)")
        ensure(not y.requires_grad, "hedef gradyan taşımamalı")
    R_.test("td_target", tgt)

    def su():
        a, b = nn.Linear(2, 2), nn.Linear(2, 2)
        a0 = [p.detach().clone() for p in a.parameters()]
        need(ns, "soft_update")(a, b, 0.1)
        for p, p0, pb in zip(a.parameters(), a0, b.parameters()):
            close(p.detach().numpy(), (0.9 * p0 + 0.1 * pb.detach()).numpy(), what="yumuşak güncelleme")
    R_.test("soft_update", su)

    def tr():
        q, lens = need(ns, "train_dqn")(30000, 0, True)
        ev = ns["evaluate"](q)
        ma = max(sum(lens[i:i + 10]) / 10 for i in range(max(1, len(lens) - 9)))
        ensure(ma >= 150 and float(ev.mean()) >= 100, f"CartPole öğrenilmeli (en iyi 10 bölüm ort. {ma:.0f}, değerlendirme {float(ev.mean()):.0f})")
        R_.table("Çift DQN, CartPole, 30 000 adım", ["ölçü", "değer"],
                 [["bölüm sayısı", len(lens)], ["ilk 10 bölüm ort. uzunluk", f"{sum(lens[:10]) / 10:.0f}"], ["en iyi 10 bölüm ort.", f"{ma:.0f}"],
                  ["açgözlü değerlendirme (20 bölüm, en fazla 500)", f"{float(ev.mean()):.0f}"]],
                 note="Rastgele politika ≈ 22 adım dayanır. DQN eğrileri gürültülüdür: performans çöküp toparlanabilir — en iyi ağı saklamak pratikte yaygındır.")
    R_.test("train_dqn", tr)
