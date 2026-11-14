"""Compute original diagrams from source task facts; never extract/reuse PDF drawings."""
from pathlib import Path
from functools import lru_cache
import sys,json
O=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/sutton-ch6');O.mkdir(parents=True,exist_ok=True)
def svg(label,w,h,key):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto"><defs><marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def text(x,y,t,size=15,anchor='middle'):return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:var(--ink)">{t}</text>'
def line(x,y,u,v,key=None,color='var(--ink)',dash=False):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="{color}" stroke-width="1.5"'+(f' marker-end="url(#{key})"' if key else '')+(' stroke-dasharray="4 3"'if dash else '')+'/>'
def node(x,y,action=False):
 fill='var(--ink)' if action else 'var(--paper)'
 return f'<circle cx="{x}" cy="{y}" r="{4 if action else 7}" fill="{fill}" stroke="var(--ink)"/>'

def figure(parts,n,caption):return '<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px">'+''.join(parts)+'</div><figcaption>Şekil '+str(n)+' — '+caption+'</figcaption></figure>'
# Driving-home example: all numbers derived rather than tracing the source chart.
time=[0,5,20,30,40,43];remaining=[30,35,15,10,3,0];total=[a+b for a,b in zip(time,remaining)];actual=[43-t for t in time];reward=[b-a for a,b in zip(time,time[1:])];mc=[a-b for a,b in zip(actual,remaining)];td=[reward[i]+remaining[i+1]-remaining[i] for i in range(5)];labels=[['ofisten','çıkış'],['araç','yağmur'],['otoyol','çıkışı'],['dar yol','kamyon'],['evin','sokağı'],['eve','varış']];parts=[]
for method in ['MC','TD']:
 key='c6-drive-'+method;b=svg(method+' toplam yolculuk tahminleri ve önerilen değişimler',350,300,key);b+=text(175,23,method+': α=1, γ=1',18);X=lambda i:40+52*i;Y=lambda v:215-(v-28)*10
 for v in [30,35,40,43,45]:b+=text(29,Y(v)+5,str(v),13,'end')+line(35,Y(v),310,Y(v),dash=True,color='var(--ink-3)')
 b+=line(35,40,35,215)+line(35,215,312,215)+text(330,75,'dk',13)
 pts=' '.join(f'{X(i)},{Y(v)}' for i,v in enumerate(total));b+=f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="2"/>'
 b+=text(265,Y(43)-7,'gerçek toplam43',13)
 for i,v in enumerate(total):
  b+=node(X(i),Y(v),True)
  if i<5:
   target=43 if method=='MC' else total[i+1]
   if target!=v:b+=line(X(i)+5,Y(v),X(i)+5,Y(target),key,'var(--c-b)')
  for j,t in enumerate(labels[i]):b+=text(X(i),238+15*j,t,12)
 b+=text(175,281,'durum / gözlem sırası',14)+'</svg>';parts.append(b)
(O/'a5.html').write_text(figure(parts,5,'Kaynak Şekil6.1,altı durum ve bütün oklar yeniden hesaplandı. MC toplam43 yönüne,TD sonraki tahmine taşır;TD farkları10,−5,5,3,0 (hesaplandı).'))
# TD backup and the five-state MRP.
b=svg('TD0 bir örnek geçiş backup;R1,gamma0.9,Vsonraki2',240,240,'c6-td');b+=text(120,25,'TD(0): durumdan duruma',16)+node(120,65)+node(120,110,True)+node(120,160)+line(120,72,120,106,'c6-td')+line(120,115,120,152,'c6-td')+text(172,100,'R=1',14)+text(120,192,'V(sonraki)=2',14)+text(120,220,'hedef=1+0.9×2=2.8',14)+'</svg>'
c=svg('Beş durumlu MRP,C başlangıç,sol0 sağ1 terminal ödülü ve doğru değerler',490,240,'c6-rw');c+=text(245,25,'Eşit olasılıklı sol/sağ yürüyüş; γ=1',17)
for i in range(7):
 x=30+i*71
 if i in [0,6]:c+=f'<rect x="{x-7}" y="90" width="14" height="14" fill="var(--ink-3)"/>'
 else:c+=node(x,97)+text(x,79,'ABCDE'[i-1])+text(x,152,f'{i}/6',16)
 if i<6:
  if i>0:c+=line(x+9,93,x+62,93,'c6-rw')
  if i<5:c+=line(x+62,101,x+9,101,'c6-rw')
  c+=text(x+35,61,'1'if i==5 else'0',14)
c+=text(243,121,'başlangıç C',15)+text(245,199,'v(i)=(v(i−1)+v(i+1))/2',17)+text(245,226,'Sınırlar0/1; çözüm v(i)=i/6',17)+'</svg>'
(O/'a6.html').write_text(figure([b,c],6,'Kaynak s.121 TD backup ve s.125 MRP,örnek geçiş ve Bellman çözümünden üretildi. MRP eylemsiz veya sabit politikanın ürettiği ödüllü zincirdir;iç komşuluklar iki yönlü,terminal uçlara yalnız giriş vardır.'))
# Predictor model; transition counts compute all values.
pb=6/8;b=svg('A kesin B,B sekiz gözlemden6kez1ve2kez0terminal;MLEmodeli',360,310,'c6-ab')
b+=node(45,130)+text(45,103,'A',22)+node(160,130)+text(160,103,'B',22)+line(54,130,150,130,'c6-ab')+text(104,112,'R=0;p=1',15)
for y,r,p in [(45,1,pb),(215,0,1-pb)]:
 b+=line(169,130,308,y,'c6-ab')+f'<rect x="310" y="{y-7}" width="14" height="14" fill="var(--ink-3)"/>'+text(260,y-18 if r else y+10,f'R={r};%{100*p:g}',16)
b+=text(180,254,'MC(A)=0; TD(A)=V(B)=6/8=0.75',16)+text(180,280,'A gözleminin MC hatası0;',15)+text(180,301,'bütün veri hatası0 değildir.',15)+'</svg>'
(O/'a7.html').write_text(figure([b],7,'Kaynak s.127 sekiz bölümünün en çok olabilirlik modeli. A→B olasılığı1;B son ödül1 olasılığı6/8. Modelde VA=VB=0.75;A için MC0. Bilinen gerçek model verilmiş değildir (hesaplandı).'))
# A state-action timeline, then three correct backup shapes.
b=svg('S,A,R,Ssonraki,Asonraki olay zinciri',340,220,'c6-events')
b+=text(170,21,'Durum–eylem çiftinden',17)+text(170,43,'sonraki çifte',17)
for i in range(4):
 x=30+(i%2)*180;y=80+(i//2)*95;suffix='+'+str(i) if i else '';b+=node(x,y)+text(x,y+26,'Sₜ'+suffix,15)+line(x+8,y,x+31,y)+node(x+35,y,True)+text(x+35,y+26,'Aₜ'+suffix,15)
 if i in [0,2]:b+=line(x+40,y,x+171,y,'c6-events')+text(x+112,y-15,f'Rₜ+{i+1}',15)
b+='<path d="M250,80 H305 V130 H10 V175 H22" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#c6-events)"/>'+text(165,124,'Rₜ+2',15)+'</svg>'
parts=['<div style="flex:1 1 100%;display:flex;justify-content:center">'+b+'</div>']
Qs=[0,1,2];epsilon=.2;gamma=.9;R=1
for method,target in [('SARSA',R+gamma*Qs[1]),('Q-öğrenme',R+gamma*max(Qs)),('Expected SARSA',R+gamma*((1-epsilon)*max(Qs)+epsilon*sum(Qs)/len(Qs)))]:
 key='c6-bk-'+method.split()[0];b=svg(method+'backup,teksonrakidurumvegelecekeylemseçimleri',235,260,key)+text(117,22,method,17)+node(117,52,True)+node(117,101)+line(117,57,117,93,key)+text(157,83,'R=1',14)
 if method=='SARSA':b+=line(117,108,117,169,key)+node(117,174,True)+text(117,204,'örnek A′:Q=1',15)
 else:
  for i,x in enumerate([42,117,192]):
   b+=line(117,108,x,169,key)+node(x,174,True)+text(x,198,f'Q={i}',14)
   if method=='Expected SARSA':b+=text(x,225,['ε/3','ε/3','1−ε+ε/3'][i],13)
  if method=='Q-öğrenme':b+='<path d="M35,160 Q117,130 199,160" fill="none" stroke="var(--ink)" stroke-width="1.6"/>'+text(117,222,'yay: maksimum',14)
 b+=text(117,251,f'hedef={target:g}; γ=0.9',15)+'</svg>';parts.append(b)
(O/'b4.html').write_text(figure(parts,4,'Kaynak s.129 olay zinciri/SARSA backup ve Şekil6.4 Q/Expected backup. Siyah eylem,açık durum düğümü;Q maksimum yayı,Expected politika ağırlıkları. Hedefler kendi Q=(0,1,2),R1,γ0.9,ε0.2 örneğinden hesaplandı;tek sonraki durum örneklenir.'))
# Full cliff map, both demonstrative paths, each crash edge.
b=svg('Dört satır12sütun uçurum,S/G,13adımlıoptimalve17adımlıörnekgüvenlirota',650,350,'c6-cliff');size=43;L=65;T=48
xy=lambda s:(L+(s%12+.5)*size,T+(s//12+.5)*size)
for s in range(48):
 x=L+s%12*size;y=T+s//12*size;fill='var(--ink-3)'if 37<=s<=46 else'var(--paper)';b+=f'<rect x="{x}" y="{y}" width="43" height="43" fill="{fill}" fill-opacity="{.4 if 37<=s<=46 else 1}" stroke="var(--ink-3)"/>'
 if s in [36,47]:b+=text(x+21,y+29,'S'if s==36 else'G',24)
optimal=[36]+list(range(24,36))+[47];safe=[36,24,12]+list(range(0,12))+[23,35,47]
assert len(optimal)-1==13 and len(safe)-1==17
for path,color in [(safe,'var(--c-a)'),(optimal,'var(--c-b)')]:b+='<polyline points="'+' '.join(','.join(map(str,xy(s)))for s in path)+'" fill="none" stroke="'+color+'" stroke-width="2.4"/>'
for s in range(37,47):
 x,y=xy(s);u,v=xy(36);b+=f'<path d="M{x},{y+7} Q{(x+u)/2},{240+(s-37)*4} {u},{v+10}" fill="none" stroke="var(--c-p)" stroke-width=".7" marker-end="url(#c6-cliff)"/>'
b+=text(325,25,'Her normal geçiş−1; uçuruma basmak−100 ve S dönüşü',17)+text(325,307,'turuncu:optimal13 / mavi:örnek güvenli17',17)+text(325,335,'Uçurum dönüşleri bölümü bitirmez.',17)+'</svg>'
(O/'b5.html').write_text(figure([b],5,'Kaynak s.132 uçurum ızgarası ve bütün başlangıca dönüşler;iki rota yasal hücre dizilerinden hesaplandı. Öğrenme eğrileri eski cliffwalk/l1,α karşılaştırması cliffsource içindedir.'))
# Maximization example with an explicit chosen 10-action implementation.
b=svg('MaksimizasyonyanlılığıMDP,A sağ0terminal/sol0B,Bdennormalödüllüçokeylem',640,330,'c6-bias')+node(410,145)+text(410,118,'A',22)+node(245,145)+text(245,118,'B',22)+line(400,145,255,145,'c6-bias')+text(327,127,'sol:R0',16)+line(419,145,565,145,'c6-bias')+text(502,127,'sağ:R0',16)+'<rect x="566" y="138" width="14" height="14" fill="var(--ink-3)"/>'
for j in range(10):
 y=35+j*23;b+=line(236,145,90,y,'c6-bias')+node(86,y,True)+line(81,y,46,y,'c6-bias')+f'<rect x="31" y="{y-5}" width="10" height="10" fill="var(--ink-3)"/>'
b+=text(166,290,'B ödülü N(−0.1,1)',18)+text(420,290,'v*(A)=0; solun değeri−0.1',17)+text(320,319,'Çizimde10 eylem bizim seçimimiz; kaynak “çok sayıda” der.',15)+'</svg>'
(O/'b6.html').write_text(figure([b],6,'Kaynak Şekil6.5 MDP alt şeması ve maxbias/l2 için kullanılan açık10-eylem seçimi. Sağın değeri0,sol−0.1;ε0.1 için doğru ε-greedy sol oranı%5. Eğriler kaynak10000 tekrar bütçesiyle yeniden örneklenir.'))
# Afterstate identity comes from applying actual legal moves.
s1=['X','','','','O','','','',''];s2=['','','','','O','X','','','']
def apply(s,a):
 assert not s[a];z=s.copy();z[a]='X';return z
out1=apply(s1,5);out2=apply(s2,0);assert out1==out2
b=svg('İkifarklıXOXdurumhamlesininaynıeylemsonrasıtahtayabirleşmesi',640,380,'c6-after')
def board(s,x,y,dotted=False):
 out='';size=27
 for k in [1,2]:out+=line(x+k*size,y,x+k*size,y+81,dash=dotted,color='var(--ink-3)')+line(x,y+k*size,x+81,y+k*size,dash=dotted,color='var(--ink-3)')
 for i,t in enumerate(s):
  if t:out+=text(x+(i%3+.5)*size,y+(i//3+.72)*size,t,23)
 return out
for s,a,x in [(s1,5,60),(s2,0,360)]:
 move=['']*9;move[a]='X';b+=board(s,x,40)+text(x+112,88,'+',24)+board(move,x+143,40,True)+line(x+107,137,300 if x==60 else 340,228,'c6-after')
b+=board(out1,280,235)+text(320,352,'Aynı afterstate; öğrenme iki çifte birlikte aktarılır.',18)+'</svg>'
(O/'b7.html').write_text(figure([b],7,'Kaynak s.137 iki farklı durum/hamle çifti,aynı X0–O4–X5 tahtasını verir. Hücre dizilerine iki yasal hamle uygulanıp eşitlik denetlendi. XOX için anlık hamle ödülü0;araba kiralamada bilinen taşıma maliyeti ayrıca tutulur (hesaplandı).'))
(O/'values.json').write_text(json.dumps(dict(backup_targets=dict(sarsa=R+gamma*Qs[1],q=R+gamma*max(Qs),expected=R+gamma*((1-epsilon)*max(Qs)+epsilon*sum(Qs)/len(Qs))),driving=dict(elapsed=time,remaining=remaining,total=total,rewards=reward,actual_remaining=actual,MC_error=mc,TD_error=td),predictor=dict(B=pb,A_TD=pb,A_MC=0,train_squared_error_sum=6*(1-pb)**2+2*pb**2),cliff=dict(optimal_path=optimal,safe_path=safe),afterstates=dict(before=[s1,s2],moves=[5,0],after=out1)),ensure_ascii=False,indent=2)+'\n')
print('7computed SVGgroups written')
