from collections import deque, Counter
from labcheck import need, ensure


class _Buf(deque):
    def __init__(self, cap):
        super().__init__()
        self.cap, self.viol = cap, 0

    def append(self, x):
        super().append(x)
        if len(self) > self.cap:
            self.viol += 1


def check(ns, R_):
    Sem, run = ns["Sem"], ns["run"]

    def t_sem():
        s = Sem(1)
        ensure(need(ns, "sem_P")(s, 0) is True and s.n == 0, "P: sayaç > 0 ise azalt ve devam")
        ensure(ns["sem_P"](s, 1) is False and list(s.q) == [1], "P: sayaç 0 ise kuyruğa gir, bloklan")
        ns["sem_P"](s, 2)
        ensure(need(ns, "sem_V")(s) == 1 and s.n == 0, "V: bekleyen varsa ilkini (FIFO) uyandır, sayaç artmaz")
        ensure(ns["sem_V"](s) == 2 and ns["sem_V"](s) is None and s.n == 1, "V: bekleyen yoksa sayaç artar")
    R_.test("semafor işlemleri", t_sem)

    def system(cap, per, prod, pills=True, seed=0):
        buf, out = _Buf(cap), []
        e, f, m, d = Sem(cap), Sem(0), Sem(1), Sem(0)
        items = [list(range(k * 100, k * 100 + per)) for k in range(3)]
        th = [prod(buf, e, f, m, it, d) for it in items] + [need(ns, "consumer")(buf, e, f, m, out) for _ in range(2)]
        if pills:
            th.append(need(ns, "closer")(buf, e, f, m, d, 3, 2))
        r = run(th, seed)
        return r, sorted(out) == sorted(sum(items, [])), buf.viol

    def bad(buf, empty, full, mutex, items, done):
        for x in items:
            yield ("P", mutex)
            yield ("P", empty)
            yield ("do", lambda x=x: buf.append(x))
            yield ("V", mutex)
            yield ("V", full)
        yield ("V", done)

    def t_sys():
        prod = need(ns, "producer")
        for s in range(150):
            r, same, viol = system(2, 6, prod, seed=s)
            ensure(r == "ok", f"tohum {s}: sistem {r} ile bitti (hepsi sonlanmalı)")
            ensure(same, f"tohum {s}: her öğe tam bir kez tüketilmeli")
            ensure(viol == 0, f"tohum {s}: tampon kapasiteyi aştı")
        rows = []
        for cap, per in ((1, 1), (4, 1), (1, 6), (4, 6)):
            a = Counter(system(cap, per, prod, seed=s)[0] for s in range(200))
            b = Counter(system(cap, per, bad, seed=s)[0] for s in range(200))
            c = Counter(system(cap, per, prod, pills=False, seed=s)[0] for s in range(200))
            rows.append([cap, 3 * per, f"{a['deadlock'] / 200:.2f}", f"{b['deadlock'] / 200:.2f}", f"{c['deadlock'] / 200:.2f}"])
        ensure(float(rows[2][3]) > 0.5 and float(rows[0][2]) == 0, "ters sıra kilitlenir, doğru sıra kilitlenmez")
        R_.table("Rastgele zamanlamalarda kilitlenme oranı (200 tohum, 3 üretici, 2 tüketici)", ["kapasite", "toplam öğe", "doğru: P(empty) → P(mutex)", "ters: P(mutex) → P(empty)", "zehirli hap yok"], rows,
                 note="Ters sırada üretici mutex'i tutarken tamponun boşalmasını bekler; tamponu boşaltacak tüketici de o mutex'i bekler: döngüsel bekleme. Tampon hiç dolmuyorsa (öğe ≤ kapasite) hata gizli kalır — testte görünmeyen, üretimde ortaya çıkan türden. Zehirli hap olmadan tüketiciler 'full' semaforunda sonsuza dek bekler: sonlandırma da bir eşzamanlama problemidir.")
    R_.test("sınırlı tampon ve sonlandırma", t_sys)
