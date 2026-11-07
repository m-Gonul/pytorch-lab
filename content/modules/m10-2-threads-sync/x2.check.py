from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        res = {}
        for v in ("correct", "turn_first", "no_turn"):
            viol, dl, n = need(ns, "peterson_states")(v)
            res[v] = (viol, dl)
            rows.append([v, n, "VAR" if viol else "yok", "VAR" if dl else "yok"])
        ensure(res["correct"] == (False, False), "Peterson: dışlama korunur, kilitlenme yok")
        ensure(res["turn_first"][0], "önce turn yazılırsa dışlama bozulur")
        ensure(res["no_turn"][1] and not res["no_turn"][0], "yalnız bayraklar: güvenli ama kilitlenebilir")
        R_.table("Peterson algoritması ve varyantları, tam durum uzayı", ["varyant", "durum", "ikisi birden kritik bölgede", "kilitlenme (ikisi de sonsuza dek bekler)"], rows,
                 note="Kanıt sıralı tutarlılık (her adım atomik ve program sırasında) varsayar. Gerçek x86'da depolama tamponu (store buffer) bayrak yazısını geciktirir; bu yüzden C++'ta seq_cst atomikler ya da çit (fence) gerekir — ödevin konusu.")
    R_.test("Peterson", t)
