import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def pc():
        ensure(need(ns, "block_params")(768) == 7087872, "block_params(768)")
        ensure(need(ns, "gpt_params")(50257, 1024, 768, 12) == 124439808, "GPT-2 small = 124,439,808")
        ensure(need(ns, "gpt_params")(65, 256, 384, 6) == 65 * 384 + 256 * 384 + 6 * (12 * 384 ** 2 + 13 * 384) + 768, "nanoGPT shakespeare-char")
    R.test("parametre sayıları", pc)
    R.test("probs: T = 1", lambda: close(need(ns, "probs")([2, 1, 0]), [0.665241, 0.244728, 0.090031]))
    R.test("probs: top-k = 2", lambda: close(need(ns, "probs")([2, 1, 0], top_k=2), [0.731059, 0.268941, 0.0]))
    R.test("probs: T = 0.5", lambda: close(need(ns, "probs")([2, 1, 0], temperature=0.5), [0.866813, 0.117310, 0.015876], atol=1e-5))
    R.test("probs: top-k sıraya bakmıyor", lambda: close(need(ns, "probs")([0, 5, 1, 4], top_k=2), [0, 0.731059, 0, 0.268941]))

    def smp():
        s = need(ns, "sample")
        ensure(s([2, 1, 0], 0.0) == 0 and s([2, 1, 0], 0.7) == 1 and s([2, 1, 0], 0.95) == 2, "ters CDF: u=0→0, 0.7→1, 0.95→2")
        ensure(s([2, 1, 0], 0.95, top_k=2) == 1, "top-k=2'de 3. token hiç seçilmez")
        ensure(s([0, 3, 1], 0.99, temperature=0) == 1, "temperature 0 → argmax")
    R.test("sample", smp)

    def gen():
        bigram = lambda toks: [3.0 if i == (toks[-1] + 1) % 5 else 0.0 for i in range(5)]
        out = need(ns, "generate")(bigram, [0], 30, temperature=0, rng=random.Random(0))
        ensure(out == [i % 5 for i in range(31)], "greedy: 0,1,2,3,4,0,…")
        out = need(ns, "generate")(bigram, [0], 2000, temperature=1.0, rng=random.Random(1))
        follow = sum(b == (a + 1) % 5 for a, b in zip(out, out[1:])) / 2000
        want = math.exp(3) / (math.exp(3) + 4)
        ensure(abs(follow - want) < 0.03, f"takip oranı {follow:.3f}, beklenen ≈ {want:.3f}")
        return f"T = 1'de 'sonraki' token oranı {follow:.3f} (teori {want:.3f})"
    R.test("generate", gen)
