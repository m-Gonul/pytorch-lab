from labcheck import need, ensure


def check(ns, R_):
    def t():
        f = need(ns, "windy_step")
        ensure(f(3, 0, 1) == (3, 1) and f(3, 3, 1) == (2, 4) and f(3, 6, 1) == (1, 7) and f(0, 6, 0) == (0, 6), "rüzgâr")
        res = [need(ns, "sarsa_windy")(8000, s) for s in range(8)]
        lens = [need(ns, "greedy_len")(Q) for Q, _ in res]
        eps = [h[-1] for _, h in res]
        ensure(all(140 <= e <= 200 for e in eps), f"8000 adımda ≈ 170 bölüm, bulunan {eps}")
        ensure(sum(l <= 18 for l in lens) >= 4 and min(lens) == 15, f"açgözlü yol ≈ 15 adım, bulunan {lens}")
        h = res[0][1]
        R_.table("Rüzgârlı ızgara, SARSA (Sutton Şekil 6.3)", ["zaman adımı", "biten bölüm"], [[t, h[t - 1]] for t in (1000, 2000, 4000, 6000, 8000)],
                 note=f"Açgözlü yol uzunlukları (8 tohum): {lens}; en kısa yol 15. 100 = açgözlü politika döngüye girdi: hiç ziyaret edilmemiş durumların Q'su 0 kaldığı için (−1 ödüllere göre iyimser) saf açgözlü yürüyüş oraya sapıp duvara takılabilir; ε-açgözlü davranış politikası bu sorunu yaşamaz. Eğri başta yatay (ilk bölüm yüzlerce adım), sonra eğimi ≈ 1/17'ye oturur.")
    R_.test("rüzgârlı ızgara", t)
