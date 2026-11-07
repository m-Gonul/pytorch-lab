from labcheck import need, ensure


def check(ns, R_):
    def t():
        n, G, B = 10000, 4, 64
        gs = need(ns, "grid_stride_indices")(n, G, B)
        ch = need(ns, "chunk_indices")(n, G, B)
        for a in (gs, ch):
            flat = sorted(i for lst in a for i in lst)
            ensure(flat == list(range(n)), "her eleman tam bir kez")
        f1 = need(ns, "coalesced_fraction")(gs, 40)
        f2 = ns["coalesced_fraction"](ch, 40)
        ensure(f1 == 1.0 and f2 == 0.0, "grid-stride birleştirilmiş, parça bölme değil")
        R_.table(f"n = {n}, {G} blok × {B} thread", ["dağıtım", "birleştirilmiş iterasyon oranı", "thread başına eleman"], [["grid-stride", f"{f1:.2f}", f"{len(gs[0])}"], ["ardışık parçalar", f"{f2:.2f}", f"{len(ch[0])}"]],
                 note="CPU thread'lerine ardışık parça vermek önbellek için iyiydi (M10.3); GPU'da tersi doğru: aynı anda çalışan warp'ın thread'leri komşu adresleri okumalı. Grid-stride döngüsü ayrıca çekirdeği herhangi bir grid boyutuyla doğru kılar.")
    R_.test("grid-stride", t)
