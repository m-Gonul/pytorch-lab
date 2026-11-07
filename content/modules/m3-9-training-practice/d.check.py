import torch
from labcheck import need, ensure


def check(ns, R_):
    def cap():
        runs = need(ns, "capacity_runs")(40, 0)
        ensure(len(runs) == 3 and all(len(v) == 40 for v in runs.values()), "üç model, 40 epoch")
        rows = []
        for name, h in runs.items():
            best = min(range(40), key=lambda i: h[i][1])
            rows.append([name, f"{h[-1][0]:.4f}", f"{h[-1][1]:.3f}", f"{h[best][1]:.3f} ({best + 1})", f"{h[-1][2]:.3f}"])
            ensure(h[-1][1] > h[best][1] * 1.05, f"{name}: doğrulama kaybı en iyiden sonra yükselmeli")
        big = list(runs.values())[-1]
        small = list(runs.values())[0]
        ensure(big[-1][1] > small[-1][1], "en büyük model en çok aşırı uyar")
        R_.table("Ders alıştırması 4 (digits ile): 40 epoch", ["mimari", "son eğitim kaybı", "son doğrulama kaybı", "en iyi doğrulama (epoch)", "son doğrulama doğruluğu"], rows,
                 note="Eğitim ve doğrulama eğrilerinin ayrışması = aşırı uyum. Büyük model en iyi noktasına daha erken ulaşır, sonra daha hızlı bozulur; doğruluk benzer kalırken kayıp kötüleşir (aşırı güvenli yanlışlar). Çare: erken durdurma, düzenleme (M3.7), daha çok veri.")
    R_.test("kapasite ve aşırı uyum", cap)

    def tr():
        keys = ("sıfırdan", "dondurulmuş", "ince ayar", "ilgisiz kaynak")
        acc = {k: 0.0 for k in keys}
        for sd in range(3):
            r = need(ns, "transfer")(100, sd)
            for k in keys:
                acc[k] += r[k] / 3
        ensure(acc["dondurulmuş"] > acc["sıfırdan"] + 0.03 and acc["ince ayar"] > acc["sıfırdan"] + 0.02, "ilgili ön-eğitim az veride kazandırmalı")
        ensure(acc["ilgisiz kaynak"] < acc["sıfırdan"], "ilgisiz kaynak zarar vermeli")
        R_.table("Hedef görev: 100 etiketli örnek (3 tohum)", ["yöntem", "test doğruluğu"], [[k, f"{v:.3f}"] for k, v in acc.items()],
                 note="Kaynak görev (gizli etkenleri tahmin etmek) hedefin ihtiyaç duyduğu gösterimi öğretir: yardımcı görevle ön-eğitim. Rastgele etiketlerle ön-eğitilen gövde ise ezberler ve kötü öznitelik üretir — aktarımın değeri kaynak ile hedefin ilişkisine bağlıdır.")
    R_.test("transfer öğrenme", tr)
