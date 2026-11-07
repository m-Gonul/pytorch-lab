from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for K in (3, 7, 21, 51):
            a = need(ns, "traffic_2d")(1024, 1024, K, 16)
            b = need(ns, "traffic_separable")(1024, 1024, K, 16)
            rows.append([K, f"{(a[0] + a[1]) / 2 ** 20:.1f}", f"{a[2] / 1e6:.0f}", f"{(b[0] + b[1]) / 2 ** 20:.1f}", f"{b[2] / 1e6:.0f}"])
        ensure(float(rows[0][1]) < float(rows[0][3]), "küçük K'de tek geçiş daha az trafik")
        R_.table("1024², uint8 girdi/çıktı, ayrılabilir ara görüntü fp32 (bayt)", ["K", "2B trafik (MiB)", "2B MAC (M)", "ayrılabilir trafik (MiB)", "ayrılabilir MAC (M)"], rows,
                 note="Ayrılabilir sürüm MAC'leri K/2 kat azaltır ama ara görüntü global bellekten bir kez gidip gelir (fp32 olarak 8 MiB). Küçük K'de çekirdek bellek-bağımlıdır ve tek geçiş kazanır; büyük K'de hesap kazancı baskındır. İki geçişi tek çekirdekte paylaşımlı bellekle birleştirmek (füzyon) ikisinin en iyisini verir.")
    R_.test("trafik modeli", t)
