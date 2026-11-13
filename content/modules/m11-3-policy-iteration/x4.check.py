import numpy as np
from labcheck import close,need,ensure

def check(ns,R_):
    def tails():
        T,r=need(ns,"location")(3,3)
        close(np.asarray(T).sum(1),np.ones(21),atol=1e-12,what="Poisson kuyruklarıyla olasılık toplamı")
        close(r[0],0,atol=1e-12,what="araba yoksa kira yok")
    R_.test("Kuyruk kütlesi korunur",tails)
    def models():
        base=need(ns,"jack_modified")(modified=False);new=ns["jack_modified"]()
        close(base["V"][0,0],421.4140633965,atol=1e-6,what="özgün 441 durumlu çözüm")
        close(new["V"][0,0],429.9463049642,atol=1e-6,what="ücretsiz taşıma + otopark")
        close(new["V"][20,20],603.5367009161,atol=1e-6,what="iki otoparkın maliyeti")
        ensure(np.asarray(new["policy"]).shape==(21,21),"tam politika boyutu")
        ensure(new["residual"]<1e-7,"Bellman artığı")
        ensure(len(base["history"])<=6 and len(new["history"])<=6,"politika yakınsaması")
        R_.table("Kaynak ve Alıştırma4.7",["model","V(0,0)","V(20,20)"],[["özgün",base["V"][0,0],base["V"][20,20]],["değişmiş",new["V"][0,0],new["V"][20,20]]])
    R_.test("İki tam model",models)
