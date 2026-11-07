import os
import time
import torch
from labcheck import need, ensure, fmt_time


def check(ns, R_):
    ext = need(ns, "build")()
    P = max(2, min(8, os.cpu_count() or 2))

    def hist():
        g = torch.Generator().manual_seed(0)
        img = (torch.randn(2048, 2048, generator=g) * 30 + 128).clamp(0, 255).to(torch.uint8)
        ref = torch.bincount(img.flatten(), minlength=256)
        rows = []
        for mode in ("single", "atomic", "mutex", "private"):
            out = ext.histogram(img, P, mode)
            ensure(torch.equal(out, ref), f"{mode}: histogram doğru olmalı")
            best = 1e9
            for _ in range(2 if mode == "mutex" else 3):
                t0 = time.perf_counter()
                ext.histogram(img, P, mode)
                best = min(best, time.perf_counter() - t0)
            rows.append([mode, fmt_time(best), "doğru"])
        flat = torch.full((2048, 2048), 7, dtype=torch.uint8)
        lost = []
        for _ in range(3):
            lost.append(int(flat.numel() - ext.histogram(flat, P, "racy")[7]))
        rows.append(["racy (tek değerli görüntü)", "", f"kayıp güncelleme: {', '.join(map(str, lost))}"])
        R_.table(f"256 kutulu histogram, 2048×2048 piksel, {P} thread", ["yöntem", "süre", "sonuç"], rows,
                 note="Atomik ve mutex doğru ama her piksel paylaşılan bir konuma dokunur: kutular az, çekişme çok. Thread'e özel histogram (PCAM: birleştirme) iletişimi 256 · p toplamaya indirir. 'racy' C++ açısından tanımlı (atomik oku, atomik yaz) ama yarış koşulu içerir; çok çekirdekte kayıp neredeyse kesindir.")
    R_.test("histogram", hist)

    def fs():
        rows = []
        for padded in (False, True):
            best = 1e9
            for _ in range(3):
                dt, cnt = ext.false_sharing(P, 2_000_000, padded)
                best = min(best, dt)
                ensure(torch.equal(cnt, torch.full((P,), 2_000_000, dtype=torch.int64)), "sayaçlar doğru")
            rows.append(["alignas(64) dolgulu" if padded else "bitişik dizi", fmt_time(best)])
        R_.table(f"{P} thread × 2 M fetch_add, her thread kendi sayacına", ["yerleşim", "süre"], rows,
                 note="Mantıksal olarak paylaşım yok, ama donanım için aynı önbellek satırı paylaşılıyor: satır çekirdekler arasında 'ping-pong' yapar.")
    R_.test("sahte paylaşım", fs)

    def bb():
        for (n, cap, pr, co) in ((10000, 4, 1, 1), (20000, 8, 3, 2), (5000, 1, 4, 4)):
            s, occ = ext.bounded_buffer(n, cap, pr, co)
            ensure(s == n * (n + 1) // 2, f"tüm öğeler bir kez tüketilmeli (n = {n})")
            ensure(1 <= occ <= cap, "doluluk kapasiteyi aşmamalı")
    R_.test("sınırlı tampon", bb)
