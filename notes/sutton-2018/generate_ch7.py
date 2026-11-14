"""Original computed diagrams for source chapter7; standard library only."""
from pathlib import Path
import sys,json,math
O=Path(sys.argv[1]);O.mkdir(exist_ok=True,parents=True)
def svg(w,h,key,aria):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{aria}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto"><defs><marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def text(x,y,t,size=15):return f'<text x="{x}" y="{y}" text-anchor="middle" style="font-size:{size}px;fill:var(--ink)">{t}</text>'
def edge(x,y,u,v,key,color='var(--ink)',dash=False):return f'<path d="M{x},{y} L{u},{v}" stroke="{color}" stroke-width="1.6" fill="none" marker-end="url(#{key})"'+(' stroke-dasharray="4 3"'if dash else'')+'/>'
def node(x,y,kind='state'):
 if kind=='terminal':return f'<rect x="{x-6}" y="{y-6}" width="12" height="12" fill="var(--ink-3)"/>'
 return f'<circle cx="{x}" cy="{y}" r="{4 if kind=="action" else 6}" stroke="var(--ink)" fill="'+('var(--ink)'if kind=='action'else'var(--paper)')+'"/>'
def fig(parts,n,cap):return '<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px">'+''.join(parts)+'</div><figcaption>Şekil '+str(n)+' — '+cap+'</figcaption></figure>'
R=[1,2,3,4];gamma=.9;leaf=2
calc=lambda n,mc=False:sum(gamma**k*R[k] for k in range(n))+(0 if mc else gamma**n*leaf)
def vpanel(title,n,key,mc=False,rhos=False,generic=False):
 b=svg(160,380,key,title+';açıkdurum,siyahörnekeylem;terminalkare')
 for i,t in enumerate(title.split('|')):b+=text(80,23+i*20,t,16)
 b+=node(80,65)
 for j in range(n):
  y=65+60*j;b+=edge(80,y+6,80,y+25,key)+node(80,y+30,'action')+edge(80,y+35,80,y+53,key)+node(80,y+60,'terminal'if mc and j==n-1 else'state')
  if rhos:b+=text(113,y+26,'ρₜ'+('₊'+str(j).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉'))if j else''),15)
 if generic or mc:b+=text(120,240,'⋮',22)
 b+=text(80,342,'örnek hedef',14)+text(80,364,f'{calc(n,mc):.4f}',15)+'</svg>';return b
parts=[]
for i,(title,n,mc,generic)in enumerate([('1-adım TD|ve TD(0)',1,False,False),('2-adım TD',2,False,False),('3-adım TD',3,False,False),('n-adım TD|örnek n=4',4,False,True),('∞-adım / MC|örnek T=4',4,True,True)]):parts.append(vpanel(title,n,'c7-v'+str(i),mc,generic=generic))
parts.append(vpanel('Özet s.157|4-adım V-IS',4,'c7-v-is',rhos=True))
(O/'a8.html').write_text(fig(parts,8,'Kaynak Şekil7.1 ve s.157 dört-adım V-IS paneli. Sonlu ufuklar açık değer yaprağında,MC terminal karede biter;ρ işaretleri state-value oranının dört eylem yarım-geçişidir. Kendi R=(1,2,3,4),γ0.9,V_yaprak2 hesabı altta gösterilir;genel n/MC için kesik devam korunur (hesaplandı).'))

def qpanel(title,n,key,mc=False,branches=(),rhos=False,sigmas=None,generic=False):
 b=svg(160,390,key,title+';eylemköklüörnekvebeklentişeması')
 for i,t in enumerate(title.split('|')):b+=text(80,23+i*20,t,16)
 b+=node(80,65,'action')
 for j in range(1,n+1):
  y=65+(j-1)*60;b+=edge(80,y+5,80,y+23,key)
  if mc and j==n:b+=node(80,y+30,'terminal');break
  b+=node(80,y+30)
  if j in branches:
   for x in [35,80,125]:b+=edge(80,y+36,x,y+54,key)+node(x,y+60,'action')
   if j==n:
    for x,t in [(35,'π₁'),(80,'π₂'),(125,'π₃')]:b+=text(x,y+81,t,13)
  else:
   b+=edge(80,y+36,80,y+55,key)+node(80,y+60,'action')
   if rhos:b+=text(105,y+50,'ρ',15)
  if sigmas is not None:b+=text(127,y+24,'σ='+str(sigmas[j-1]),13)
 if generic or mc:b+=text(130,258,'⋮',22)
 b+=text(80,365,'siyah: eylem',13)+'</svg>';return b
parts=[]
for i,(title,n,mc,generic)in enumerate([('1-adım SARSA|SARSA(0)',1,False,False),('2-adım SARSA',2,False,False),('3-adım SARSA',3,False,False),('n-adım SARSA|örnek n=4',4,False,True),('∞-adım / MC|örnek T=4',4,True,True)]):parts.append(qpanel(title,n,'c7-q'+str(i),mc=mc,generic=generic))
parts.append(qpanel('n-adım Expected|örnek n=4',4,'c7-q-es',branches=(4,)))
(O/'b8.html').write_text(fig(parts,8,'Kaynak Şekil7.3. SARSA çift/eylem kökünden örnek eylem yaprağına gider;MC terminale ulaşır. Expected son durumdan bütün eylemlereπ ağırlığıyla açılır;üç eylem temsili çizildi. S/A dizileri ve sonlanma tipleri algoritmanın kayıtlarından oluşturuldu.'))
# Actual source grid geometry, then actual positive action-value increments.
path=[(4,1),(4,2),(4,3),(3,3),(3,4),(2,4),(1,4),(1,5),(1,6),(1,7),(2,7),(3,7),(3,8),(4,8),(5,8),(5,7),(5,6),(4,6)]
assert len(path)==18 and all(abs(a-c)+abs(b-d)==1 for(a,b),(c,d)in zip(path,path[1:]))
T=len(path)-1;panels=[];changes={}
for i,(title,n)in enumerate([('Görülen 17 hamle',None),('1-adım SARSA',1),('10-adım SARSA',10)]):
 key='c7-grid'+str(i);b=svg(280,320,key,title+';8satır10sütun,kaynakrotasıvehesaplananQartışları')+text(140,25,title,17);size=24;L=20;top=60;xy=lambda z:(L+(z[1]+.5)*size,top+(z[0]+.5)*size)
 for row in range(8):
  for col in range(10):b+=f'<rect x="{L+col*size}" y="{top+row*size}" width="{size}" height="{size}" fill="var(--paper)" stroke="var(--ink-3)" stroke-width=".7"/>'
 if n is None:
  points=[xy(z)for z in path];points[-1]=(points[-1][0],points[-1][1]+9)
  b+='<polyline points="'+' '.join(','.join(map(str,z))for z in points)+'" fill="none" stroke="var(--c-a)" stroke-width="2" marker-end="url(#'+key+')"/>'
 else:
  Q={};active=[]
  for tau in range(T):
   h=min(tau+n,T);G=1 if h==T else Q.get((path[h],path[h+1]),0);pair=(path[tau],path[tau+1]);Q[pair]=Q.get(pair,0)+.5*(G-Q.get(pair,0))
   if Q[pair]>0:active.append(tau)
  assert active==list(range(T-min(n,T),T));changes[str(n)]={'indices':active,'Q':[Q[(path[k],path[k+1])]for k in active]}
  for tau in active:
   x,y=xy(path[tau]);u,v=xy(path[tau+1]);dx=(u-x)/size;dy=(v-y)/size;b+=edge(x-dx*6,y-dy*6,x+dx*8,y+dy*8,key,'var(--c-a)')
 sx,sy=xy(path[0]);gx,gy=xy(path[-1]);b+=text(gx,gy+5,'G',17)+text(140,282,'Q0;yalnız G ödülü+1',15)+text(140,304,'kendi ölçeğimiz α=0.5',14)+'</svg>';panels.append(b)
(O/'b9.html').write_text(fig(panels,9,'Kaynak Şekil7.4 geometrisi hücre dizisine kodlanıp yasal17 geçişle denetlendi. Yalnız pozitif Q değişimleri okla çizilir:tek-adımda1,on-adımda10 eylem. Kaynak pozitif ödülün büyüklüğünü vermediğinden+1 veα0.5 kendi açık ölçeğimizdir (hesaplandı).'))
# Seven leaf probabilities sum to one before discount factors.
pi=[.2,.5,.3];leaves=[[1,3],[2,4],[0,2,5]];qrows=[[1,4,3],[2,1,4],[0,2,5]];rewards=[1,-1,2];G=2+.9*sum(p*q for p,q in zip(pi,qrows[2]))
G=-1+.9*(.2*2+.3*4+.5*G);G=1+.9*(.2*1+.3*3+.5*G)
leafweights=[.2,.3,.1,.15,.05,.125,.075];assert abs(sum(leafweights)-1)<1e-15
b=svg(340,560,'c7-tree','ÜçadımTree;7yaprak,örnekS/A/Romurgasıveπçarpımları')+text(170,24,'3-adım: bütün yapraklardan',17)+node(170,52,'action')+text(224,56,'Sₜ,Aₜ',14)
for j in range(3):
 y=105+j*130;prev=52 if j==0 else y-75;b+=edge(170,prev+5,170,y-7,'c7-tree')+node(170,y)+text(224,y+5,'Sₜ₊'+str(j+1).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')),14)+text(220,y-22,'Rₜ₊'+str(j+1).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉'))+'='+str(rewards[j]),14)
 for a,x in enumerate([55,170,285]):
  b+=edge(170,y+6,x,y+48,'c7-tree')+node(x,y+54,'action')
  if a==1 and j<2:b+=text(x-55,y+75,'örnek Aₜ₊'+str(j+1).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')),13)
  else:
   q=qrows[j][a];weight=(.5**j)*pi[a];b+=text(x,y+77,'Q='+str(q),14)+text(x,y+96,'πürünü='+f'{weight:g}',13)
b+=text(170,503,'γ=0.9; hedef='+f'{G:.6f}',17)+text(170,530,'iç eylem Q değerleri doğrudan toplanmaz',14)+'</svg>'
(O/'b10.html').write_text(fig([b],10,'Kaynak s.152 numarasız3-adım tree-backup. Üç örnek durum,iki ara örnek eylem ve3 ödül,yanlarda7 değer yaprağı. π=(0.2,0.5,0.3) için gösterilen yaprak çarpımları1 eder;ödül/indirim ayrıca uygulanıp hedef3.048625 hesaplandı. Kendi sayısal değerlerimizdir;genel algoritma tüm eylemleri toplar.'))
parts=[qpanel('4-adım|SARSA',4,'c7-sigma-sarsa',rhos=True),qpanel('4-adım|Tree backup',4,'c7-sigma-tree',branches=(1,2,3,4)),qpanel('4-adım|Expected SARSA',4,'c7-sigma-es',branches=(4,),rhos=True),qpanel('4-adım|Q(σ)',4,'c7-sigma-mix',branches=(2,4),rhos=True,sigmas=[1,0,1,0])]
(O/'b11.html').write_text(fig(parts,11,'Kaynak Şekil7.5 ve s.157 Q(σ) özet paneli. ρ yalnız politika dışı örnek eylem yarım-geçişlerinde;Tree bütün seviyelerde,Expected son seviyede,karışık şema2/4 seviyelerinde beklenti dalları açar. Üç yaprak temsili,generic n yerine burada dört adım. Kontrol-değişkenli7.17 ile düzSARSA arasındaki cebirsel ayrım metinde açıklanır.'))
(O/'values.json').write_text(json.dumps(dict(V_targets={str(n):calc(n)for n in [1,2,3,4]},MC_target=calc(4,True),grid_path=path,positive_Q_changes=changes,tree=dict(target=G,leaf_weights=leafweights,pi=pi,rewards=rewards,Q=qrows)),indent=2)+'\n');print('5 computed SVG groups written')
