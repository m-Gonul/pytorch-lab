import numpy as np
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera().astype(np.float64)
    M, N = img.shape
    y, x = np.mgrid[:M, :N]
    noise = 40 * np.sin(2 * np.pi * (30 * y / M + 50 * x / N)) + 30 * np.sin(2 * np.pi * (-70 * y / M + 20 * x / N))
    noisy = img + noise

    def sp():
        got = set(need(ns, "find_spikes")(noisy, 4))
        ensure(got == {(30, 50), (-30, -50), (-70, 20), (70, -20)}, f"tepeler: {sorted(got)}")
    R.test("find_spikes (iki sinüs = dört tepe)", sp)

    def nr():
        H = need(ns, "notch_reject")((64, 64), [(5, 3)], 2)
        ensure(H[32 + 5, 32 + 3] == 0 and H[32 - 5, 32 - 3] == 0, "çentik merkezleri sıfır")
        close(H[0, 0], 1.0, atol=0.01, what="uzakta ≈ 1")
    R.test("notch_reject", nr)

    def rp():
        out = need(ns, "remove_periodic")(noisy, 2, 3)
        P = lambda a: 10 * np.log10(255 ** 2 / np.mean((a - img) ** 2))
        ensure(P(out) > P(noisy) + 15, f"PSNR {P(noisy):.1f} → {P(out):.1f}")
        return f"PSNR {P(noisy):.1f} → {P(out):.1f} dB"
    R.test("periyodik gürültü temizleme", rp)
