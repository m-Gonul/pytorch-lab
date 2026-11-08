import time
import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        lo = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
        close(need(ns, "dice_loss")(lo, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])).item(), 0.0, atol=1e-3, what="mükemmel tahmin")
        close(need(ns, "seg_metrics")(torch.tensor([1.0, 1, 0, 0]), torch.tensor([1.0, 0, 1, 0])), (0.5, 0.5, 0.5), what="metrikler")
        Xt, Yt = ns["lesions"](300, 1)
        rows, R = [], {}
        t0 = time.perf_counter()
        for loss in ("bce", "wbce", "dice", "bce+dice"):
            m = need(ns, "train_seg")(loss, 300, 0)
            with torch.no_grad():
                P = (m(Xt) > 0).float()
            R[loss] = ns["seg_metrics"](P, Yt)
            rows.append([loss.upper(), f"{R[loss][0]:.3f}", f"{R[loss][1]:.3f}", f"{R[loss][2]:.3f}"])
        ensure(R["wbce"][1] > R["bce"][1] and R["wbce"][2] < R["bce"][2], "ağırlıklı BCE: duyarlılık ↑, kesinlik ↓")
        ensure(R["dice"][1] > R["bce"][1] and max(R["dice"][0], R["bce+dice"][0]) > R["bce"][0] - 0.01, "Dice duyarlılığı artırır")
        R_.table(f"Küçük lezyonlar (ön plan ≈ %{100 * float(Yt.mean()):.0f}), 300 adım × 4 kayıp ({time.perf_counter() - t0:.0f} sn)", ["kayıp", "Dice", "duyarlılık (recall)", "kesinlik (precision)"], rows,
                 note="BCE piksel başına ortalandığı için baskın arka planı doğru bilmek yeter: kenar ve küçük lezyon pikselleri kaçırılır (yüksek kesinlik, düşük duyarlılık). Ön plan ağırlığı duyarlılığı yükseltip yanlış pozitif artırır. Dice kaybı ön plan örtüşmesini doğrudan optimize eder; BCE + Dice (nnU-Net varsayılanı) kalibre olasılık ve iyi örtüşmeyi birleştirir. Hangisinin 'iyi' olduğu klinik maliyete bağlıdır: lezyon kaçırmak mı daha pahalı, yanlış alarm mı?")
    R_.test("segmentasyon kayıpları", t)
