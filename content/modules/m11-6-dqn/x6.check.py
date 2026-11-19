from labcheck import close, need, ensure
import numpy as np


def check(ns, R_):
    def t():
        tiles = need(ns, "source_tiles")
        for x,v in [(-1.2,-.07),(-.5,0.),(.5,.07),(-.4,.02)]:
            ids = tiles(x,v,2)
            ensure(len(ids)==8 and len(set(ids))==8 and min(ids)>=0 and max(ids)<2640,"sekiz ayrı/geçerli aktif karo")
        out = need(ns, "mountaincar_source")(episodes=5,n=1,alpha=.5,seed=1010001)
        # Filled from the independent direct one-step Python/C++ reference.
        expected = [2954, 455, 402, 345, 375]
        close(out["lengths"], expected, what="bağımsız doğrudan bir-adım referansı")
        ensure(not out["cutoff"] and out["actual_steps"]==sum(out["lengths"]),"yalnız gerçek tamamlanmış bölümler")
        cut = ns["mountaincar_source"](episodes=2,n=1,budget=10)
        ensure(cut["cutoff"] and cut["lengths"]==[] and cut["actual_steps"]==10 and cut["unfinished_steps"]==10,"kesme terminal veya başarılı bölüm değildir")
        ensure(np.isfinite(out["w"]).all(),"sonlu ağırlıklar")
        R_.table("Beş gerçek bölüm",["n", "uzunluklar", "gerçek geçiş"],[[1,str(out["lengths"]),out["actual_steps"]]])
    R_.test("Negatif tile koordinatları, güncelleme sırası ve bütçe",t)
