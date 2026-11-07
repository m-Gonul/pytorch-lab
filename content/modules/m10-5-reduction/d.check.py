import torch
from labcheck import close, need, ensure


def check(ns, R_):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)

    def sums():
        for n in (1, 1000, 1024, 5000, 300000):
            x = torch.randn(n, generator=g).to(dev)
            ref = float(x.double().sum())
            close(float(need(ns, "sum_two_pass")(x)), ref, atol=1e-3 * max(1, abs(ref)) + 1e-3, what=f"iki geçiş n = {n}")
            close(float(need(ns, "sum_atomic")(x)), ref, atol=1e-3 * max(1, abs(ref)) + 1e-3, what=f"atomik n = {n}")
            ensure(need(ns, "argmax")(x) == int(torch.argmax(x)), f"argmax n = {n}")
        ensure(need(ns, "block_sums")(torch.ones(3000).to(dev)).tolist() == [1024.0, 1024.0, 952.0], "blok toplamları")
    R_.test("toplam ve argmax", sums)

    def hist():
        x = torch.randint(0, 50, (20000,), generator=g, dtype=torch.int32).to(dev)
        ensure(torch.equal(need(ns, "histogram")(x, 50).cpu(), torch.bincount(x.cpu(), minlength=50).int()), "histogram")
        y = torch.full((5000,), 7, dtype=torch.int32).to(dev)
        ensure(int(ns["histogram"](y, 10)[7]) == 5000, "tek değerli girdi: atomikler hiçbirini kaybetmemeli")
    R_.test("histogram", hist)

    def bf():
        x = torch.randn(10, 32, generator=g, dtype=torch.float64)
        out = need(ns, "butterfly")(x)
        close(out.numpy(), x.sum(1, keepdim=True).expand(10, 32).numpy(), atol=1e-12, what="her şerit warp toplamını tutar")
        R_.note("Kelebek: 5 adımda her şerit (lane) toplamı görür — paylaşımlı bellek ve __syncthreads olmadan. __shfl_down_sync ise yalnız şerit 0'da sonuç bırakır.")
    R_.test("butterfly (__shfl_xor_sync)", bf)
