from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        f = need(ns, "semi_gradient_n")
        # Unequal intermediate rewards, discount and same-group revisits catch
        # frozen targets, wrong powers and omitted terminal flushing.
        ep = [(100, 1.0, 200), (200, 2.0, 900), (900, 3.0, None)]
        close(f([ep], .2, 10, 2, .5), [1.14, .6], what="MC: G=(2.75,3.5,3); ortak grup güncellemeleri sıralı")
        close(f([ep], .2, 1, 2, .5), [.56, .6], what="TD0: tek geçişli ödül/terminal hedefi")
        close(f([ep], .2, 2, 2, .5), [1.02, .6], what="iki adım: G0=2, G1=3.5; terminal kuyruğu")
        close(f([ep, ep], .2, 1, 2, .5), [1.0232, 1.08], what="ikinci bölüm: güncel ağırlıkla bootstrap")
        close(f([], .2, 2), [0.0] * 20, what="boş veri")
        try:
            f([ep], .2, 0)
        except ValueError:
            pass
        else:
            ensure(False, "n=0 reddedilmeli")
        R_.table("Aynı üç geçiş, γ=.5, α=.2", ["n", "w1", "w2"], [[n, *f([ep], .2, n, 2, .5)] for n in (1, 2, 3, 10)])
    R_.test("İndirim, ortak grup ve terminal kuyruğu", t)
