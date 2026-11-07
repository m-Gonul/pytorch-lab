from labcheck import close, no_imports, need, ensure


def conv(k, ci, co, g=1):
    return {"type": "conv", "k": k, "cin": ci, "cout": co, "groups": g, "bias": True}


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    P = {"type": "pool", "k": 2}

    def ex5():
        cnn2 = [conv(7, 1, 16), P, conv(3, 16, 32), P, conv(3, 32, 64), P, {"type": "flatten"}, {"type": "linear", "in": 576, "out": 64}, {"type": "linear", "in": 64, "out": 10}]
        fc3 = [{"type": "flatten"}, {"type": "linear", "in": 784, "out": 74}, {"type": "linear", "in": 74, "out": 50}, {"type": "linear", "in": 50, "out": 10}]
        fc1 = [{"type": "flatten"}, {"type": "linear", "in": 784, "out": 300}, {"type": "linear", "in": 300, "out": 100}, {"type": "linear", "in": 100, "out": 10}]
        bn = lambda c: {"type": "bn", "c": c}
        cnn4 = [conv(7, 1, 64), bn(64), P, bn(64), conv(5, 64, 128), bn(128), conv(3, 128, 128), bn(128), P, bn(128), conv(3, 128, 256), bn(256), conv(3, 256, 256), bn(256), P,
                {"type": "flatten"}, {"type": "linear", "in": 2304, "out": 128}, {"type": "linear", "in": 128, "out": 64}, {"type": "linear", "in": 64, "out": 10}]
        rows = []
        for name, m in (("FC 784-300-100-10", fc1), ("CNN (alıştırma 2)", cnn2), ("FC 784-74-50-10", fc3), ("BN'li büyük CNN (alıştırma 4)", cnn4)):
            p, mac, shp = need(ns, "count")(m, (1, 28, 28))
            rows.append([name, f"{p:,}".replace(",", " "), f"{mac / 1e6:.2f} M"])
        ensure(ns["count"](cnn2, (1, 28, 28))[0] == 61514 and ns["count"](fc3, (1, 28, 28))[0] == 62350 and ns["count"](cnn4, (1, 28, 28))[0] == 1546954, "parametre sayıları")
        ensure(ns["count"](cnn2[:4], (1, 28, 28))[2] == (32, 7, 7), "çıkış şekli")
        R_.table("Ders alıştırması 5'in modelleri", ["model", "parametre", "MAC/örnek"], rows,
                 note="Eşit parametreli FC ve CNN'de hesap çok farklı: CNN her ağırlığı yüzlerce konumda yeniden kullanır (5.7 M MAC vs 0.06 M). Parametre ≠ hesap.")
    R_.test("sayaçlar", ex5)

    def blocks():
        p, c = need(ns, "inception_params")(192, 64, 96, 128, 16, 32, 32)
        ensure(p == 192 * 64 + 192 * 96 + 9 * 96 * 128 + 192 * 16 + 25 * 16 * 32 + 192 * 32 and c == 256, "Inception 3a")
        dw = need(ns, "count")([conv(3, 128, 128, 128), conv(1, 128, 256)], (128, 14, 14))
        st = ns["count"]([conv(3, 128, 256)], (128, 14, 14))
        ensure(dw[0] < st[0] / 8, "ayrılabilir ~9× az parametre")
        close(need(ns, "lrn")([0, 0, 2.0, 2.0, 0], 2, 5, 1e-4, 0.75)[2], 2 / (2 + 1e-4 * 8) ** 0.75, what="LRN")
    R_.test("bloklar ve LRN", blocks)
