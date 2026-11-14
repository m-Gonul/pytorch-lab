"""Own chapter8 SVGs from actual measurements. No source artwork is embedded.
python notes/sutton-2018/generate_ch8.py /tmp/ch8-figures
"""
from pathlib import Path
import json,math,sys,html,re
from collections import deque
from functools import lru_cache
BASE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True,parents=True)
D=json.loads((BASE/'measurements_ch8_js.json').read_text());TR=json.loads((BASE/'measurements_ch8_trajectory.json').read_text())
C=['var(--c-a)','var(--c-b)','var(--c-p)','var(--c-q)'];INK='var(--ink)'
def svg(w,h,key,aria):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(aria,quote=True)}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto"><defs><marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def txt(x,y,s,size=14,anchor='middle',color=INK):return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color}">{html.escape(str(s))}</text>'
def line(x,y,u,v,color=INK,dash=False,width=1.5,arrow=None):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 3"'if dash else'')+(f' marker-end="url(#{arrow})"'if arrow else'')+'/>'
def box(x,y,w,h,label,color='var(--paper)'):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" stroke="var(--ink-3)" fill="{color}"/>'+txt(x+w/2,y+h/2+5,label)
def node(x,y,kind='s',color=INK):
 if kind=='terminal':return f'<rect x="{x-5}" y="{y-5}" width="10" height="10" stroke="{color}" fill="var(--paper)"/>'
 return f'<circle cx="{x}" cy="{y}" r="{3 if kind=="a" else 5}" stroke="{color}" fill="'+(color if kind=='a'else'var(--paper)')+'"/>'
def fig(n,parts,cap):
 cap=re.sub(r'(?<=[a-zA-ZçğıöşüÇĞİÖŞÜ])(?=\d)', ' ',cap);cap=re.sub(r'(?<=\d)(?=[a-zA-ZçğıöşüÇĞİÖŞÜ])',' ',cap);cap=re.sub(r';(?=\S)','; ',cap);cap=re.sub(r',(?=\S)',', ',cap)
 s='<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px">'+''.join(parts)+'</div><figcaption>Şekil '+str(n+1)+' — '+cap+'</figcaption></figure>'
 (OUT/f'f{n}.html').write_text(s)
 return s

def plot(title,series,xlabel,ylabel,key,xlim=None,ylim=None,log=False,marks=(),xticks=None,yticks=None):
 w,h=330,330;L,R,T,B=46,317,54,245
 xs=[x for z in series for x in z['x']];ys=[y for z in series for y in z['y']if y is not None and math.isfinite(y)]
 xmin,xmax=xlim or(min(xs),max(xs));ymin,ymax=ylim or(min(0,min(ys)),max(ys)*1.05)
 if log:ymin,ymax=10,max(1e7,ymax)
 X=lambda x:L+(x-xmin)/(xmax-xmin)*(R-L)
 Y=lambda y:B-(math.log10(y/ymin)/math.log10(ymax/ymin)if log else(y-ymin)/(ymax-ymin))*(B-T)
 b=svg(w,h,key,title)+txt(w/2,22,title,15)+txt(L,43,ylabel,12,'start')+line(L,T,L,B)+line(L,B,R,B)
 yt=yticks or ([10**j for j in range(int(math.log10(ymin)),int(math.log10(ymax))+1)]if log else[ymin+(ymax-ymin)*j/4 for j in range(5)])
 for y in yt:
  label=('10'+str(int(math.log10(y))).translate(str.maketrans('0123456789','⁰¹²³⁴⁵⁶⁷⁸⁹')))if log else (f'{y:.0f}' if ymax>=10 else f'{y:.2g}')
  b+=line(L,Y(y),R,Y(y),'var(--ink-3)',True,.5)+txt(L-5,Y(y)+4,label,11,'end')
 for x,label in (xticks or[(xmin,f'{xmin:g}'),((xmin+xmax)/2,f'{(xmin+xmax)/2:g}'),(xmax,f'{xmax:g}')]):b+=txt(X(x),B+17,label,11)
 for x,label in marks:b+=line(X(x),T,X(x),B,'var(--ink-3)',True)+txt(X(x),T+14,label,10)
 for z in series:
  path='';started=False
  for x,y in zip(z['x'],z['y']):
   if y is None or not math.isfinite(y)or(log and y<=0):started=False;continue
   path+=(' L'if started else' M')+f'{X(x):.3f},{Y(y):.3f}';started=True
  b+=f'<path d="{path}" stroke="{z.get("color",C[0])}" stroke-width="1.8" fill="none"'+(' stroke-dasharray="5 3"'if z.get('dash')else'')+'/>'
 for j,z in enumerate(series):
  yy=281+j*15
  if yy>h-3:break
  b+=line(15,yy-4,35,yy-4,z.get('color',C[0]),z.get('dash',False),2)+txt(42,yy,z['name'],11,'start')
 b+=txt((L+R)/2,272,xlabel,12)
 return b+'</svg>'

# Source p160's two chains, p162 triangle, and Fig8.1 architecture.
parts=[]
b=svg(320,380,'c8-chain','Planlama tanımı ve ortak model-deneyim-yedek-değer-politika zinciri')+txt(160,23,'Model → politika',17)
b+=box(20,48,95,35,'Model')+box(205,48,95,35,'Politika')+line(120,65,200,65,arrow='c8-chain')+txt(160,54,'planlama',12)
for i,label in enumerate(['Model','Simüle deneyim','Yedek hedefi','Değerler','Politika']):
 y=115+i*48;b+=box(77,y,166,31,label)
 if i<4:b+=line(160,y+32,160,y+46,arrow='c8-chain')
b+=txt(160,373,'DP de bu zincire uyar',13);parts.append(b+'</svg>')
b=svg(320,380,'c8-triangle','Deneyim doğrudan RL ve model üzerinden planlama ile değerleri iyileştirir')+txt(160,23,'Gerçek deneyimin iki rolü',17)
b+=box(92,61,136,38,'Değer / politika')+box(16,275,112,38,'Model')+box(190,275,114,38,'Deneyim')
b+=line(190,270,79,270,arrow='c8-triangle')+txt(155,258,'model öğrenme',12)
b+=line(76,270,136,106,arrow='c8-triangle')+txt(74,178,'planlama',12)
b+=line(227,270,183,106,arrow='c8-triangle')+txt(246,182,'doğrudan RL',12)
b+=line(233,100,294,252,arrow='c8-triangle')+txt(272,128,'eylem',12)+txt(160,356,'Dolaylı yol modelden geçer',13);parts.append(b+'</svg>')
b=svg(320,380,'c8-dyna','Dyna mimarisi: gerçek ve simüle deneyimin aynı değer güncellemesine bağlanması')+txt(160,23,'Dyna mimarisi — Şekil8.1',17)
for x,y,w,label in [(56,53,208,'Politika / değer'),(20,170,111,'Gerçek deneyim'),(180,170,122,'Simüle deneyim'),(20,302,111,'Çevre'),(191,302,111,'Model')]:b+=box(x,y,w,34,label)
b+=line(72,92,72,166,arrow='c8-dyna')+txt(48,128,'eylem',11)
b+=line(56,299,56,209,arrow='c8-dyna')+line(42,167,94,91,arrow='c8-dyna')+txt(35,105,'doğrudan',11)+txt(28,119,'RL',11)
b+=line(247,298,247,208,arrow='c8-dyna')+txt(274,240,'arama',11)+txt(274,253,'denetimi',11)
b+=line(242,165,215,91,arrow='c8-dyna')+txt(265,126,'planlama',11)+txt(265,139,'yedeği',11)
b+=line(137,190,218,297,arrow='c8-dyna')+txt(155,247,'model',11)+txt(154,261,'öğrenme',11)
b+=line(75,208,75,298,arrow='c8-dyna')+txt(160,366,'Ortak son yol: aynı RL yedeği',13);parts.append(b+'</svg>')
fig(2,parts,'Kaynak s.160–163: iki numarasız zincir, numarasız etkileşim üçgeni ve Şekil8.1. Bütün bilgi akışları, doğrudan ve model üzerinden yollar bağımsız düzenle çizildi. Kavramsal olarak süreçler paralel; uygulama seri olabilir.')

WALL={(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7)}
def grid(title,key,kind=0,changed=False,Q=None,agent=None):
 b=svg(300,270,key,title)+txt(150,23,title,15);size=27;L,top=25,46
 for r in range(6):
  for c in range(9):
   wall=(r,c)in WALL if kind==0 else r==3 and((c>=1 if changed else c<=7)if kind==1 else c>=1 and(c<=7 if changed else True))
   x,y=L+c*size,top+r*size;b+=f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="'+('var(--ink-3)'if wall else'var(--paper)')+'" stroke="var(--ink-3)" stroke-width=".7"/>'
   if Q is not None and not wall and(r,c)!=(0,8):
    row=Q[4*(r*9+c):4*(r*9+c+1)]
    if min(row)!=max(row):
     for a in range(4):
      if row[a]==max(row):
       dr,dc=[(-1,0),(1,0),(0,-1),(0,1)][a];b+=line(x+size/2-dc*4,y+size/2-dr*4,x+size/2+dc*8,y+size/2+dr*8,C[0],width=1.4,arrow=key)
 sr,sc=(2,0)if kind==0 else(5,3);b+=txt(L+(sc+.5)*size,top+(sr+.5)*size+5,'S',15)+txt(L+8.5*size,top+.5*size+5,'G',15)
 if agent is not None:
  r,c=divmod(agent,9);b+=f'<rect x="{L+c*size+10}" y="{top+r*size+10}" width="7" height="7" fill="var(--ink)"/>'
 b+=txt(150,232,'↑ ↓ ← →; sınırda / duvarda öz döngü',12)+txt(150,254,'Koordinatlar sıfırdan; 6×9',12)
 return b+'</svg>'
parts=[grid('47 durum; en kısa yol14','c8-maze')]
series=[dict(name=f'n={z["n"]}',x=list(range(2,51)),y=z['mean'][1:],color=C[i])for i,z in enumerate(D['static'])]
parts.append(plot('30 koşu ×50 tam bölüm',series,'bölüm','bölüm başına adım','c8-dynacurve',xlim=(2,50),ylim=(14,800),xticks=[(x,str(x))for x in[2,10,20,30,40,50]],yticks=[14,200,400,600,800]))
for z in D['snapshots']:
 tr=z['trace'];parts.append(grid(f'n={z["n"]}: ikinci bölüm ortası','c8-pol'+str(z['n']),Q=tr['Q'],agent=tr['s']))
fig(3,parts,'Kaynak Şekil8.2/8.3. Eğri gerçek bağımsız deneyden; ilk bölüm bütün n için736.3333 adım olduğu için kaynak gibi grafikte dışarıda. İkinci bölümün gerçek yarı anındaki Q eşitlikleri okların yokluğunu, siyah kare ajanın yerini belirler. Kaynağın kare konumlarını sayısal tohumla taklit etmedik (hesaplandı).')
parts=[]
for kind in [1,2]:
 parts +=[grid(('Engelleme'if kind==1 else'Kısayol')+': önce','c8-old'+str(kind),kind),grid('Değişimden sonra','c8-new'+str(kind),kind,True)]
 series=[]
 for j,z in enumerate([x for x in D['changing']if x['parameters']['kind']==kind]):
  ix=list(range(0,len(z['mean']),50))+[len(z['mean'])-1];series.append(dict(name=['Dyna-Q','Dyna-Q+','Yalnız seçim bonusu'][j],x=[i+1 for i in ix],y=[z['mean'][i]for i in ix],color=C[j]))
 parts.append(plot(('Engelleme'if kind==1 else'Kısayol')+':30 koşu',series,'gerçek adım','birikimli gerçek ödül','c8-change'+str(kind),xlim=(0,3000 if kind==1 else 6000),ylim=(0,200 if kind==1 else 400),xticks=[(x,str(x))for x in([0,1000,2000,3000]if kind==1 else[0,3000,6000])],marks=[(1000 if kind==1 else 3000,'değişim')]))
fig(4,parts,'Kaynak Şekil8.4/8.5 ve Alıştırma8.4. Dört harita, S/G ve değişim zamanları korunur; eğriler n50/α1/κ.001 ile kendi30 koşumuzdan. Bonus gerçek ödülde sayılmaz. Kaynak ek deneyin n/κ/koşu sayısını belirtmez; üçüncü eğri bizim çözdüğümüz seçim-bonusu deneyidir (hesaplandı).')

series=[];censored=[]
for method in [0,1]:
 entries=sorted([z for z in D['priority']if z['method']==method],key=lambda z:z['scale']);values=[]
 for z in entries:
  avg=sum(r['updates']for r in z['results'])/len(z['results']);ok=all(r['found']for r in z['results']);values.append(avg if ok else None)
  if not ok:censored.append(dict(states=z['results'][0]['states'],mean_updates_at_limit=avg,successes=sum(r['found']for r in z['results'])))
 series.append(dict(name='Dyna-Q'if not method else'PS; θ=10⁻¹²',x=list(range(8)),y=values,color=C[method]))
b=plot('Büyüyen ızgara;5 tohum',series,'durum sayısı (47×2ᵏ)','optimal başlangıç yoluna kadar yedek','c8-scaling',log=True,xticks=[(i,str(47*2**i))for i in range(8)])
fig(5,[b],'Kaynak s.170 numarasız ölçek grafiğinin bağımsız yeniden deneyi. Kaynakta47/94/186/376/752/1504/3008/6016 işaretleri; bizim hücre çoğaltmamız47×2ᵏ. Dyna doğrudan yedekleri dahil n5,α.5,γ.95. PS1504 noktasında4/5 başarı; bir koşu2m gerçek adımda bitmediği için tamamlanma ortalaması çizilmiyor. Diğer noktalar5/5 başarı. Eski θ10⁻⁴ koşuları da veride saklı; kaynakta bildirilen5–10× her ölçekte zorunlu değil (hesaplandı).')

# Own rod model: four endpoint-valid motions, two positive translations, ±10deg rotation.
obstacles=[(5.5,9,8.5,13),(10,3.5,12.5,7),(3,3.5,5,6),(12.5,12,15.5,15),(15,8.5,17,10),(10,9,12,12)]
half=2.2
@lru_cache(None)
def valid(x,y,a):
 theta=a*math.pi/18;dx,dy=half*math.cos(theta),half*math.sin(theta);p=(x-dx,y-dy);q=(x+dx,y+dy)
 if min(p[0],q[0])<-.5 or max(p[0],q[0])>19.5 or min(p[1],q[1])<-.5 or max(p[1],q[1])>19.5:return False
 for xmin,ymin,xmax,ymax in obstacles:
  lo,hi=0.,1.
  for start,delta,low,high in [(p[0],q[0]-p[0],xmin,xmax),(p[1],q[1]-p[1],ymin,ymax)]:
   if abs(delta)<1e-12:
    if low<=start<=high:continue
    lo,hi=1.,0.;break
   u,v=sorted(((low-start)/delta,(high-start)/delta));lo=max(lo,u);hi=min(hi,v)
  if lo<=hi:return False
 return True
start=(3,17,27);goal=(17,3,27)
def successors(s):
 x,y,a=s;theta=a*math.pi/18;dx,dy=round(math.cos(theta)),round(math.sin(theta))
 candidates=[(x+dx,y+dy,a),(x-dy,y+dx,a),(x,y,(a+1)%36),(x,y,(a-1)%36)]
 return [z for z in candidates if 0<=z[0]<20 and 0<=z[1]<20 and valid(*z)]
parents={start:None};queue=deque([start])
while queue:
 s=queue.popleft()
 if s==goal:break
 for z in successors(s):
  if z not in parents:parents[z]=s;queue.append(z)
assert goal in parents,'own rod model must have a legal solution'
path=[];s=goal
while s is not None:path.append(s);s=parents[s]
path.reverse();assert all(b in successors(a)for a,b in zip(path,path[1:]))
b=svg(330,430,'c8-rod','Engelli alanda dört eylemli çubuğun BFS ile bulunan en kısa rotası')+txt(165,23,'Çubuk: kendi açık modelimiz',16)
L,T,size=25,48,14
b+=f'<rect x="{L-7}" y="{T-7}" width="280" height="280" fill="var(--paper)" stroke="var(--ink-3)"/>'
for xmin,ymin,xmax,ymax in obstacles:b+=f'<rect x="{L+xmin*size}" y="{T+ymin*size}" width="{(xmax-xmin)*size}" height="{(ymax-ymin)*size}" fill="var(--ink-3)"/>'
for i,(x,y,a)in enumerate(path):
 theta=a*math.pi/18;dx,dy=half*math.cos(theta)*size,half*math.sin(theta)*size;colour=C[1]if i==0 else C[2]if i==len(path)-1 else C[0]
 b+=line(L+x*size-dx,T+y*size-dy,L+x*size+dx,T+y*size+dy,colour,width=2.5 if i in[0,len(path)-1]else.6)
 b+=f'<rect x="{L+x*size-1.7}" y="{T+y*size-1.7}" width="3.4" height="3.4" stroke="{colour}" fill="var(--paper)"/>'
b+=txt(65,315,'S',16,color=C[1])+txt(278,75,'G',16,color=C[2])+txt(165,352,f'BFS: {len(path)-1} yasal hareket',15)+txt(165,376,'20×20 merkez;36 yön;4 eylem',13)+txt(165,400,'Mavi: ara çubuklar ve merkezler',13)
fig(6,[b+'</svg>'],'Kaynak s.171 numarasız manevra çiziminin amacını kendi hesaplanan haritamızla gösteriyoruz. Çubuk uzunluğu4.4 hücre; iki pozitif öteleme: yuvarlanmış uzun eksen ve ona+90° dik eksen, iki dönme±10°. Çarpışma modeli hareket sonundaki çizgi parçası/engel kesişimini tam denetler; süpürülen ara hareketi modellemeyen ayrık bir görevdir. Kaynağın kesin geometrisi veya aynı optimal yolu iddia edilmez (hesaplandı).')

# All seven one-step backup diagrams, including the deliberately empty eighth cell.
parts=[]
for row,(name,q,optimal)in enumerate([('Vπ',False,False),('V*',False,True),('Qπ',True,False),('Q*',True,True)]):
 for sample in [False,True]:
  key=f'c8-backup{row}{int(sample)}';b=svg(155,270,key,name+(' örnek'if sample else' beklenen'))+txt(77,20,name,17)+txt(77,40,'örnek'if sample else'beklenen',13)
  if row==1 and sample:b+=txt(77,130,'Bu hücre boş',13)+txt(77,151,'Tek örnek max',12)+txt(77,169,'yerine geçmez',12);parts.append(b+'</svg>');continue
  top=65;b+=node(77,top,'a'if q else's')
  if not q:
   ax=[77]if sample else[35,77,119]
   for x in ax:
    b+=line(77,top+6,x,104)+node(x,108,'a')
    sx=[x]if sample else[x-12,x+12]
    for z in sx:b+=line(x,111,z,159)+node(z,165)
   b+=txt(28,94,'max'if optimal else('A'if sample else'π'),11)+txt(137,137,'R'if sample else'p,r',11)
  else:
   sx=[77]if sample else[35,119]
   for x in sx:
    b+=line(77,top+4,x,108)+node(x,114)
    ax=[x]if sample and not optimal else[x-18,x+18]
    for z in ax:b+=line(x,120,z,165)+node(z,171,'a')
   b+=txt(29,90,'R'if sample else'p,r',11)+txt(77,197,'max'if optimal else('A′'if sample else'π'),12)
  title=(['Politika değerlendirme','Değer yinelemesi','Q-politika değerlendirme','Q-değer yinelemesi'][row]if not sample else['TD(0)','','SARSA','Q-öğrenme'][row])
  words=title.split(' ')
  for j,t in enumerate(words):b+=txt(77,224+j*17,t,12)
  parts.append(b+'</svg>')
fig(7,parts,'Kaynak Şekil8.6: V/Q × π/* × beklenen/örnek; yedi algoritma ve boş v* örnek hücresi. Açık daire durum, siyah nokta eylem; p/π ağırlıkları, R örneği ve max ayrımı korunur. Dallar temsili iki/üç seçenek; aynı işlem daha çok dala genişler.')
series=[]
for j,b in enumerate([2,10,100,1000,10000]):
 xx=list(range(1,2*b+1))if b<100 else sorted(set([1]+[max(1,round(2*b*i/150))for i in range(1,151)]));series.append(dict(name=f'b={b}',x=[t/b for t in xx],y=[math.sqrt((b-1)/(b*t))for t in xx],color=C[j%4],dash=j==4))
# Plot two panels to keep every family label readable on mobile.
parts=[plot('Örnek RMS: b2/10/100',series[:3],'hesap / b','normalize RMS','c8-rms-a',xlim=(0,2),ylim=(0,1)),plot('Örnek RMS: b1000/10000',[*series[3:],dict(name='beklenen',x=[0,1,1,2],y=[1,1,0,0],color=INK)],'hesap / b','normalize RMS','c8-rms-b',xlim=(0,2),ylim=(0,1))]
fig(8,parts,'Kaynak Şekil8.7. Beş b eğrisi aynı0–2b/0–1 ölçeğinde iki okunabilir panele ayrıldı; tam beklenen yedek b hesap sonunda0. Örnek eğrileri σ²=(b−1)/b,α1/t koşuluyla analitik RMS hesabıdır. b/eş olasılık/ilk hata1 tek başına bu formülü belirlemez (hesaplandı).')
parts=[]
for c in TR['cases']:
 parts.append(plot(f'S={c["S"]}, b={c["b"]};200 görev',[dict(name='döngüsel uniform',x=c['points'],y=c['mean'][0],color=C[0],dash=True),dict(name='ε.1 yörünge',x=c['points'],y=c['mean'][1],color=C[1])],'beklenen yedek','gerçek açgözlü v(0)',f'c8-traj{c["S"]}{c["b"]}',ylim=(-.2,9)))
fig(9,parts,"Kaynak Şekil8.8'in üst üç b ve alt10000/b1 deneyi; Alıştırma8.8'in alt10000/b3 ek deneyi. Bütün panellerin y ölçeği aynı; gerçek politika değerleri184 milyon beklenen yedekten. Kaynağın≈3.5 b1 yüksekliğinden farklı kendi model/tohum sonuçlarıdır; eğri uydurulmadı. Standart hatalar l6'da açılabilir; kaynak ile kendi tercihleri anlatımda ayrılır (hesaplandı).")

b=svg(330,260,'c8-relevance','Başlangıç kümesi ilgili durumların, ilgili durumlar tüm durumların içinde')+txt(165,24,'Optimal kısmi politika',16)
b+='<ellipse cx="165" cy="130" rx="146" ry="75" fill="var(--ink-3)" stroke="var(--ink)"/><ellipse cx="135" cy="136" rx="96" ry="49" fill="var(--paper)" stroke="var(--ink)"/><ellipse cx="86" cy="130" rx="23" ry="16" fill="var(--c-a)" stroke="var(--ink)"/>'
b+=txt(86,132,'S₀',12)+txt(151,143,'İlgili',14)+txt(238,90,'İlgisiz',14)+txt(165,224,'İlgili: bazı optimal politikayla erişilir',12)+txt(165,246,'İlgisiz: hiçbir optimal politikayla erişilmez',12)
fig(10,[b+'</svg>'],'Kaynak s.177 numarasız küme diyagramı: başlangıç, ilgili ve hiçbir optimal yoldan erişilmeyen ilgisiz durumlar. Kendi47 durumlu RTDP deneyinde bazı optimal yollardaki18 durum BFS uzaklıklarıyla ayrıca hesaplandı; bu18 sayısı9115 durumlu kaynak pistine ait değildir (hesaplandı).')

# A selective look-ahead: 3 left and 7 right postorder backups, then root action selection.
leaves=iter([1,4,2,3,-1,2,6,1,3,0,5,4]);backups=[]
def tree(depth):
 if depth==0:return dict(value=float(next(leaves)))
 children=[tree(depth-1),tree(depth-1)];item=dict(children=children,value=max(x['value']for x in children),id=len(backups)+1);backups.append(item);return item
left=tree(2);right=tree(3);assert len(backups)==10
parts=[]
def draw(t,xlo,xhi,y):
 x=(xlo+xhi)/2;t['x']=x;t['y']=y
 if 'children'in t:
  for i,ch in enumerate(t['children']):
   lo=xlo+(xhi-xlo)*i/2;hi=xlo+(xhi-xlo)*(i+1)/2;z=(lo+hi)/2;ay=y+21;bodies.append(line(x,y+6,z,ay)+node(z,ay,'a')+line(z,ay+3,z,y+65));draw(ch,lo,hi,y+72)
  bodies.append(f'<ellipse cx="{x}" cy="{y+24}" rx="{(xhi-xlo)/2-3}" ry="32" fill="none" stroke="var(--c-a)" stroke-width="1"/>');bodies.append(node(x,y)+txt(x+11,y-10,str(t['id']),12,'start',C[0])+txt(x,y+53,f'V={t["value"]:g}',12))
 else:bodies.append(node(x,y)+txt(x,y+22,f'{t["value"]:g}',13))
b=svg(320,180,'c8-root','Kökte iki aday eylem;sol3ve sağ10numaralı alt ağaç sonuçları')+txt(160,22,'Önce 1…10, sonra kök seçimi',16)+node(160,48)
b+=line(160,54,73,83)+node(71,87,'a')+line(71,91,71,121)+node(71,127)+line(160,54,247,83)+node(249,87,'a')+line(249,91,249,121)+node(249,127)
b+=txt(71,151,'Sol: yedek3 → 4',13)+txt(249,151,'Sağ: yedek10 → 6',13)+txt(160,175,'Kök eylemi: sağ',13)
parts.append(b+'</svg>')
for t,w,key,title in [(left,300,'c8-search-left','Sol alt ağaç: 1,2,3'),(right,360,'c8-search-right','Sağ alt ağaç: 4…10')]:
 bodies=[];draw(t,8,w-8,57)
 h=300 if t is left else 370
 b=svg(w,h,key,title)+txt(w/2,22,title,16)+''.join(bodies)+txt(w/2,h-16,'r=0,γ=1; açık yapraklar kendi V tahminimiz',12)
 parts.append(b+'</svg>')
fig(11,parts,'Kaynak Şekil8.9: on alt ağaç yedeği postorder1..10 ile hesaplandı; kökte iki eylem karşılaştırılır. Mobilde okunabilmesi için kök, sol ve sağ alt ağaç ayrı panellerde. Durum/eylem düğümleri, seçici farklı derinlik, mavi çevreler ve numaralı sıra korunur. Kendi deterministik modelimizde beklenti tek ardıldır; yaprak kestirimleri1,4,2,3,−1,2,6,1,3,0,5,4 → kök eylem değerleri4/6 (hesaplandı).')

# One actual MC trajectory and its prefix-sensitive returns.
rewards=[1,-1,4];g=.9;G2=4;G1=-1+g*G2;G0=1+g*G1
parts=[]
for i,title in enumerate(['1 Seçim','2 Genişletme','3 Simülasyon','4 Yedekleme']):
 key='c8-mcts'+str(i);b=svg(165,405,key,title+';tek örnek turun kayıtları')+txt(82,23,title,15)
 coords=[(82,65),(43,133),(81,212)]
 for j,(x,y)in enumerate(coords):
  if j==2 and i==0:continue
  b+=node(x,y,color=C[0]if j==2 and i>=1 else INK)+txt(x+21,y,'S'+str(j),12,'start')
 if i>=0:b+=line(80,71,50,103,C[0],width=2,arrow=key)+node(48,108,'a')+line(46,112,43,126,C[0],width=2,arrow=key)
 b+=line(87,70,125,106,'var(--ink-3)')+node(128,111,'a')
 if i>=1:b+=line(46,138,70,176,C[0],width=2,arrow=key)+node(73,181,'a')+line(75,185,81,205,C[0],width=2,arrow=key)
 if i>=2:b+=line(82,218,82,291,C[0],True,2,arrow=key)+node(82,254,'a')+'<path d="M76,303 L88,303 L82,291 Z" fill="var(--paper)" stroke="var(--ink)"/>'
 if i==3:b+=line(98,287,100,222,C[1],width=2,arrow=key)+line(104,207,62,142,C[1],width=2,arrow=key)+line(59,128,99,71,C[1],width=2,arrow=key)
 b+=txt(82,331,'r=(1,−1,4),γ=.9',12)
 if i<3:b+=txt(82,354,'İçeride: ağaç politikası',11)+txt(82,373,'Dışarıda: rollout',11)
 else:b+=txt(82,353,f'G(S₁,a₁)={G1:g}',12)+txt(82,373,f'G(S₀,a₀)={G0:g}',12)
 parts.append(b+'</svg>')
fig(12,parts,'Kaynak Şekil8.10: zaman kaldıkça dört faz yeniden yapılır. Yeni S2 genişletilir; onun sonrasındaki kesikli rollout ve terminal üçgeni kalıcı ağaç Q girdisi üretmez. Gerçek örnek getiri kökte3.34, ikinci kenarda2.6; prefix ödülleri yüzünden aynı sayıyı her kenara yazmak yanlış olur. Önceden kök N3/Q2 ise yeni N4/Q2.335; ikinci kenar N1/Q1 ise N2/Q1.8 (hesaplandı).')

b=svg(440,510,'c8-dimensions','Yedek genişliği ve derinliği;TD DP MC ve kapsamlı arama köşeleri')
b+='<defs><linearGradient id="c8-grad" x1="0" x2="1" y1="0" y2="1"><stop offset="0%" stop-color="var(--c-a)" stop-opacity=".15"/><stop offset="100%" stop-color="var(--c-p)" stop-opacity=".7"/></linearGradient></defs><rect x="95" y="94" width="250" height="250" fill="url(#c8-grad)" stroke="var(--ink-3)"/>'
b+=line(95,62,345,62,C[1],width=2,arrow='c8-dimensions')+txt(220,45,'Genişlik: örnek → beklenen',14)+line(65,100,65,342,C[2],width=2,arrow='c8-dimensions')+txt(64,366,'Derinlik',13)+txt(220,212,'n adım, karma yedekler',14)+txt(220,236,'örnek + dağılım',13)
for x,y,label,full,width in [(95,95,'TD',False,False),(345,95,'DP',False,True),(95,344,'MC',True,False),(345,344,'Tam arama',True,True)]:
 b+=txt(x,y-15 if y<200 else y+115,label,14)+node(x,y)
 positions=[x-27,x+27]if width else[x]
 for u in positions:
  b+=line(x,y+5,u,y+24)+node(u,y+28,'a')
  vv=[u-14,u+14]if width else[u]
  for v in vv:b+=line(u,y+31,v,y+52)+node(v,y+57)
  if full:b+=line(u,y+61,u,y+80,INK,True)+node(u,y+86,'terminal')
b+=txt(230,500,'On / off politika: sayfaya dik üçüncü eksen',13)
fig(13,[b+'</svg>'],'Kaynak Şekil8.11: genişlik yatay, derinlik dikey; TD/DP/MC/tam arama köşeleri, terminal kareleri, kesikli uzun devam ve karma yöntemlerin iç alanı. Gölgeli renk alanı bir performans metriği değildir. λ karışımları sol kenar; seçici sezgisel arama sağ kenar; üçüncü on/off ekseni çizim düzlemine dik düşünülür.')

values=dict(rod=dict(obstacles=obstacles,half_length=half,start=start,goal=goal,shortest_steps=len(path)-1,path=path,endpoint_collision_only=True),search=dict(leaf_values=[1,4,2,3,-1,2,6,1,3,0,5,4],postorder=[dict(id=x['id'],value=x['value'])for x in backups],root_action_values=[left['value'],right['value']]),mcts=dict(rewards=rewards,gamma=g,returns=[G0,G1,G2],new_root_Q=(3*2+G0)/4,new_second_Q=(1+G1)/2),priority_censored=censored)
(OUT/'values.json').write_text(json.dumps(values,indent=2)+'\n')
# Keep the browser's fixed full-budget data mechanically tied to the provenance JSON.
p=BASE.parents[1]/'site/js/labs/trajectoryplan.js';s=p.read_text();s=re.sub(r'/\* CH8_DATA_BEGIN \*/.*?/\* CH8_DATA_END \*/','/* CH8_DATA_BEGIN */ '+json.dumps(TR['cases'],separators=(',',':'))+' /* CH8_DATA_END */',s,flags=re.S);p.write_text(s)
print('12 computed SVG groups and rod/search/MCTS records written',values['rod']['shortest_steps'])
