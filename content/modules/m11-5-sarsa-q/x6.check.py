import numpy as np
from labcheck import need,ensure


def check(ns,R_):
    def env():
        step=need(ns,'windy_step')
        ensure(step(38,3)==27,'hedefin sağında sola gitmek hedefin bir üstüne götürür; kaynak sütun')
        class Noise:
            def __init__(self,z):self.z=z
            def integers(self,*args):return self.z
        ensure([step(38,3,True,Noise(z)) for z in (-1,0,1)]==[37,27,17],'pozitif rüzgâr üç durum ve1/3 olasılık')
        ensure(step(30,8)==30,'bekleme yalnız rüzgâr kadar hareket eder')
        ensure(all(step(30,1,True,Noise(z))==31 for z in (-1,0,1)),'rüzgâr0 sütunlarına rastgele rüzgâr ekleme')
    def model():
        for actions,dist in [(4,15),(8,7),(9,7)]:
            V,pi=need(ns,'optimal_windy')(actions,False)
            ensure(abs(V[30]+dist)<1e-9,'bağımsız model minimum adım')
            path,done=need(ns,'trajectory')(pi)
            ensure(done and len(path)-1==dist,'optimal model yörüngesi')
        V,pi=need(ns,'optimal_windy')(8,True)
        ensure(abs(V[30]+10.9646090534948)<1e-8,'stokastik sekiz eylem model referansı')
    def learn():
        for actions,stochastic in [(8,False),(9,False),(8,True)]:
            out=need(ns,'sarsa_windy')(4000,actions,stochastic,.5,.1,1)
            ensure(out['Q'].shape==(70,actions) and np.isfinite(out['Q']).all(),'Q boyutu/sonluluk')
            ensure(sum(out['lengths'])+out['unfinished']==4000,'tam bölümler ve bitmemiş parça toplamı')
            ensure(len(out['history'])==4000 and np.all(np.diff(out['history'])>=0),'adım başına bölüm sayacı')
            ensure(out['counts'].sum()==4000,'her TD adımında tek güncelleme')
            ensure((out['Q'][37]==0).all(),'terminal Q0')
    R_.test('Kaynak rüzgâr fiziği',env)
    R_.test('4/8/9 eylem ve stokastik optimum',model)
    R_.test('King hamleleri ve stokastik SARSA',learn)
