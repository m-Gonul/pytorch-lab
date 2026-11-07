import torch
from labcheck import close, need, ensure, fmt_time


def check(ns, R_):
    def t():
        g = torch.Generator().manual_seed(1)
        img = torch.rand(40, 50, generator=g)
        k = torch.rand(7, 7, generator=g)
        ref = need(ns, "conv_direct")(img, k)
        close(need(ns, "conv_unfold")(img, k).numpy(), ref.numpy(), atol=1e-4, what="unfold + matmul")
        close(need(ns, "conv_fft")(img, k).numpy(), ref.numpy(), atol=1e-3, what="FFT")
        res = need(ns, "crossover")(256, [3, 7, 15, 31, 51])
        ensure(len(res) == 5 and all(r[4] < 1e-4 for r in res), "üç yöntem aynı sonucu vermeli")
        ensure(res[-1][3] < res[-1][1], "K = 51'de FFT doğrudan konvolüsyondan hızlı olmalı")
        R_.table("256×256, tek kanal (CPU)", ["K", "F.conv2d", "unfold + matmul", "FFT", "göreli fark"], [[K, fmt_time(a), fmt_time(b), fmt_time(c), f"{d:.0e}"] for K, a, b, c, d in res],
                 note="Doğrudan ve im2col yöntemlerin maliyeti K² ile, FFT'ninki neredeyse sabit. Kesişim noktası donanıma ve kütüphaneye bağlı; cuDNN her katman için birkaç algoritmayı deneyip en hızlısını seçer (torch.backends.cudnn.benchmark = True).")
    R_.test("yöntem kesişimi", t)
