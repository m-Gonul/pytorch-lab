import torch
from labcheck import close, max_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    for f in ("quantize_int4", "pack_int4", "unpack_int4", "dequant_int4"):
        R.test(f"{f} döngüsüz", (lambda f=f: max_loops(src, 0, func=f)))
    g = torch.Generator().manual_seed(0)
    W = torch.randn(64, 256, generator=g)

    def rt():
        q = torch.randint(-8, 8, (5, 12), generator=g, dtype=torch.int8)
        p = need(ns, "pack_int4")(q)
        ensure(p.dtype == torch.uint8 and tuple(p.shape) == (5, 6), "uint8, yarı genişlik")
        ensure(torch.equal(need(ns, "unpack_int4")(p), q), "pack → unpack özdeşlik (negatifler dahil)")
        ensure(int(p[0, 0]) == ((int(q[0, 0]) & 0xF) | ((int(q[0, 1]) & 0xF) << 4)), "düşük 4 bit çift indeks, yüksek 4 bit tek indeks")
    R.test("pack/unpack int4", rt)

    def qz():
        q, s = need(ns, "quantize_int4")(W, 32)
        ensure(q.dtype == torch.int8 and int(q.min()) >= -8 and int(q.max()) <= 7, "q ∈ [-8, 7]")
        ensure(tuple(s.shape) == (64, 8), "ölçek (out, in/group)")
        Wd = need(ns, "dequant_int4")(ns["pack_int4"](q), s, 32)
        err = (Wd - W).abs().reshape(64, 8, 32)
        ensure((err <= s[..., None] / 2 + 1e-6).all(), "grup içi hata ≤ s/2")
    R.test("quantize_int4 + dequant", qz)

    def table():
        rows = []
        Wo = W.clone(); Wo[:, 7] *= 30
        for gsz in (256, 128, 32):
            q, s = ns["quantize_int4"](Wo, gsz)
            Wd = ns["dequant_int4"](ns["pack_int4"](q), s, gsz)
            rel = ((Wd - Wo).norm() / Wo.norm()).item()
            bits = 4 + 16 / gsz
            rows.append([str(gsz), f"{rel:.3f}", f"{bits:.2f}"])
        rels = [float(r[1]) for r in rows]
        ensure(rels[2] < rels[0], "küçük grup daha az hata vermeli")
        R.table("int4, aykırı sütunlu W (64 × 256)", ["grup", "göreli hata", "bit/ağırlık (fp16 ölçek dahil)"], rows,
                note="Grup küçüldükçe aykırı değerin etkisi o gruba hapsolur; bedeli ölçekler için ek bit.")
    R.test("grup boyutu ve hata", table)
