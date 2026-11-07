import numpy as np
from scipy import signal
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))

    def hp():
        H = need(ns, "lowpass")("butterworth", (64, 80), 10, 2)
        close(H[32, 40], 1.0, what="merkez"); close(H[42, 40], 0.5, what="D = D0")
        close(need(ns, "lowpass")("gaussian", (64, 80), 10)[32, 50], np.exp(-0.5), what="Gauss D = D0")
        I = ns["lowpass"]("ideal", (64, 80), 10)
        ensure(I[32, 50] == 1 and I[32, 51] == 0, "ideal sınır dahil")
        close(need(ns, "highpass")("gaussian", (64, 80), 10), 1 - ns["lowpass"]("gaussian", (64, 80), 10), what="yüksek geçiren")
    R.test("süzgeç tasarımı", hp)

    def ff():
        img = data.camera()[::2, ::2].astype(np.float64)
        M, N = img.shape
        H = ns["lowpass"]("gaussian", (2 * M, 2 * N), 25)
        got = need(ns, "freq_filter")(img, H)
        fp = np.zeros((2 * M, 2 * N)); fp[:M, :N] = img
        want = np.real(np.fft.ifft2(np.fft.ifftshift(H) * np.fft.fft2(fp)))[:M, :N]
        close(got, want, atol=1e-8, what="Gonzalez hattı = fftshift'li hat")
    R.test("freq_filter", ff)

    def cf():
        rng = np.random.default_rng(0)
        img = rng.random((60, 70)); k = rng.random((9, 5))
        close(need(ns, "conv_fft")(img, k), signal.convolve2d(img, k, "full"), atol=1e-9, what="tam konvolüsyon")
    R.test("conv_fft = scipy convolve2d (full)", cf)

    def ring():
        img = np.zeros((128, 128)); img[:, 64:] = 200
        H = lambda kind: ns["lowpass"](kind, (256, 256), 15, 2)
        rows = []
        over = {}
        for kind in ("ideal", "butterworth", "gaussian"):
            out = ns["freq_filter"](img, H(kind))[64, 20:110]
            over[kind] = max(out.max() - 200, 0)
            rows.append([kind, f"{out.max():.1f}", f"{over[kind]:.1f}"])
        ensure(over["ideal"] > over["butterworth"] > over["gaussian"] - 1e-9, f"aşma: {over}")
        R.table("Basamak kenarı, D0 = 15", ["süzgeç", "en büyük değer (basamak 200)", "aşma"], rows, note="İdeal süzgecin sinc yanıtı halkalanır; n = 2 Butterworth hafifçe; Gauss hiç.")
    R.test("halkalanma: ideal > Butterworth > Gauss", ring)
