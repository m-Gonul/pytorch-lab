"""Own chapter9 figures from simulated data, formulas and computed feature overlaps.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python generate_ch9.py OUT
No source pixels or curves are copied. See README.md for choices and seeds.
"""
from pathlib import Path
import sys,json,html,math,re,base64,io
import numpy as np
from PIL import Image
from scipy.spatial import Voronoi,voronoi_plot_2d
BASE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
D=json.loads((BASE/'measurements_ch9_walk.json').read_text())
E=json.loads((BASE/'measurements_ch9_exact.json').read_text())
C=['var(--c-a)','var(--c-b)','var(--c-p)','var(--c-q)'];INK='var(--ink)'
def svg(w,h,key,aria):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(aria,quote=True)}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto"><defs><marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def txt(x,y,s,size=14,anchor='middle',color=INK):
 s=re.sub(r'(?<=[A-Za-zÇĞİÖŞÜçğıöşü])(?=\d)', ' ',str(s));s=re.sub(r'(?<=\d)(?=[A-Za-zÇĞİÖŞÜçğıöşü])', ' ',s);s=re.sub(r':(?=\S)', ': ',s)
 return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color}">{html.escape(str(s))}</text>'
def line(x,y,u,v,color=INK,dash=False,width=1.5,arrow=None):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 3"'if dash else'')+(f' marker-end="url(#{arrow})"'if arrow else'')+'/>'
def box(x,y,w,h,label,color='var(--paper)'):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" stroke="var(--ink-3)" fill="{color}"/>'+txt(x+w/2,y+h/2+5,label)
def node(x,y,kind='s',color=INK):
 if kind=='terminal':return f'<rect x="{x-5}" y="{y-5}" width="10" height="10" stroke="{color}" fill="var(--paper)"/>'
 return f'<circle cx="{x}" cy="{y}" r="{3 if kind=="a" else 5}" stroke="{color}" fill="'+(color if kind=='a'else'var(--paper)')+'"/>'
def fig(n,parts,cap):
 cap=re.sub(r'(?<=[a-zA-ZçğıöşüÇĞİÖŞÜ])(?=\d)', ' ',cap);cap=re.sub(r'(?<=\d)(?=[a-zA-ZçğıöşüÇĞİÖŞÜ])',' ',cap);cap=re.sub(r';(?=\S)','; ',cap);cap=re.sub(r',(?=\S)',', ',cap)
 s='<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px">'+''.join(parts)+'</div><figcaption>Şekil '+str(n)+' — '+cap+'</figcaption></figure>'
 (OUT/f'f{n}.html').write_text(s)
 return s

def plot(title,series,xlabel,ylabel,key,xlim=None,ylim=None,log=False,marks=(),xticks=None,yticks=None):
 w,h=330,365;L,R,T,B=46,317,54,245
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
  yy=300+j*15
  if yy>h-3:break
  b+=line(15,yy-4,35,yy-4,z.get('color',C[0]),z.get('dash',False),2)+txt(42,yy,z['name'],11,'start')
 b+=txt((L+R)/2,280,xlabel,12)
 return b+'</svg>'

def image_matrix(a,x,y,w,h):
 a=np.asarray(a);pix=np.uint8(np.clip(a,0,1)*255);buf=io.BytesIO();Image.fromarray(pix).save(buf,format='PNG')
 return f'<image href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" style="image-rendering:pixelated"/>'
def heat(title,a,key,axis=True,plus=None):
 b=svg(220,252,key,title)+txt(110,20,title,14)+image_matrix(a,30,35,174,174)
 if axis:b+=txt(117,239,'s₁ →',13)+txt(15,126,'s₂',12)+txt(30,225,'0',11)+txt(203,225,'1',11)+txt(21,43,'1',11)
 if plus is not None:
  x,y=plus;b+=line(x-4,y,x+4,y,width=2)+line(x,y-4,x,y+4,width=2)
 return b+'</svg>'

x=list(range(1,1001));grp=E['groups']['10']
fig(2,[plot('10 grup: MC ve TD',[
 dict(name='gerçek vπ',x=x,y=E['V'],color=C[0]),
 dict(name='MC: 100000 bölüm',x=x,y=[D['mc']['weights'][(s-1)//100]for s in x],color=C[1]),
 dict(name='TD popülasyon noktası',x=x,y=[grp['TD_weights'][(s-1)//100]for s in x],color=C[2],dash=True)],'durum','değer','c9-walk',ylim=(-1,1),xticks=[(1,'1'),(500,'500'),(1000,'1000')]),
 plot('Ziyaret ağırlığı µ', [dict(name='tam η / toplam η',x=x,y=E['mu'],color=C[0])],'durum','µ(s)','c9-mu',ylim=(0,.014),yticks=[0,.0017,.007,.0137],xticks=[(1,'1'),(500,'500'),(1000,'1000')])],
 'Şekil9.1 ve Şekil9.2sol karşılığı:100000 gerçek MC bölümü/α0.00002 ile bulunan merdiven,1000×1000 Bellman denkleminin tam çözümü ve bağımsız TD moment çözümü. µ, başlangıç500 için η=h+Pᵀη çözümünden hesaplandı. Uçlar±0.92196;µ(500)=0.01369235, beklenen bölüm83.15245 adım. TD çizgisi asimptotik moment çözümüdür, sonlu α örnek koşusu değildir. Kaynak eğrileri kopyalanmadı (hesaplandı/simüle edildi).')
ns=D['nstep'];parts=[]
for k in range(2):
 series=[dict(name='n='+str(ns['n'][j]),x=ns['alpha'],y=ns['results']['mean'][j*21:j*21+21],color=C[j%4],dash=j%5==4)for j in range(k*5,k*5+5)]
 parts.append(plot('n-adım: '+('1–16'if k==0 else'32–512'),series,'α','ilk10 bölüm RMS','c9-n'+str(k),ylim=(.25,.85)))
fig(3,parts,'Şekil9.2sağ:20grup×50durum,100bağımsız koşu×10tam bölüm, n=1..512 ve α=0,.05,..1.0. Her bölüm sonrası1000durumun ağırlıksız RMS hatası,sonra10bölüm ve100koşu ortalaması. Aynı koşuda bütün yöntemler aynı yolları gördü. Seçilen tohumlarda en düşük ortalama n4/α.4 için0.2870553; evrensel en iyi n iddiası değildir. µ-ağırlıklı VE ile karıştırılmamalı (simüle edildi).')
xx=np.linspace(0,1,201)
fig(4,[plot(f'cos({i}πs)',[dict(name=f'i={i}',x=xx,y=np.cos(i*np.pi*xx),color=C[(i-1)%4])],'s','özellik','c9-cos'+str(i),ylim=(-1,1))for i in range(1,5)],'Şekil9.3: ilk dört değişken kosinüs özelliği. i0 ayrıca sabit1 özelliğidir; s0..1 normalize edilmiştir (formülden hesaplandı).')
X,Y=np.meshgrid(np.linspace(0,1,100),np.linspace(1,0,100));coeff=[(0,1),(1,0),(1,1),(0,5),(2,5),(5,2)]
fig(5,[heat(f'c=({a},{b})',(1+np.cos(np.pi*(a*X+b*Y)))/2,f'c9-f{j}')for j,(a,b)in enumerate(coeff)],'Şekil9.4: altı iki-boyutlu Fourier özelliği cos(πcᵀs). Beyaz+1/siyah−1. Sıfır katsayılı boyut sabittir; iki pozitif katsayı birlikte etkileşim taşır. Dördüncü panelin görünür kaynak etiketi(0,5); metin çıkarımındaki(1,5) görsel etiketi değildir (hesaplandı).')
base=D['bases'];parts=[]
for i in range(2):
 parts.append(plot('Polinom'if i==0 else'Fourier',[dict(name=c['name'],x=base['checkpoints'],y=c['results']['mean'],color=C[j])for j,c in enumerate(base['curves'][3*i:3*i+3])],'bölüm','kök µ-VE',f'c9-b{i}',ylim=(0,.5)))
fig(6,parts,'Şekil9.5:30koşu×5000tam bölüm, mertebe5/10/20; her mertebede n+1özellik. Polinomα.0001, Fourierα.00005 her özellikte sabit; kaynak figürünün belirtmediği frekans normalizasyonu uygulanmadı. Son ortalamalar polinom.125817/.119514/.114755 ve Fourier.053819/.055414/.056857. Sonuç, basit monomların bu görevde yavaş kaldığını gösterir; tüm polinom ailelerini karşılaştırmaz (simüle edildi).')
# Circular fields: S and S' share exactly one field.
centers=np.array([[.28,.45],[.42,.28],[.52,.5],[.7,.35],[.73,.68],[.27,.77]])
S=np.array([.35,.48]);SP=np.array([.63,.52]);rad=.24
active=np.linalg.norm(centers-S,axis=1)<rad;activep=np.linalg.norm(centers-SP,axis=1)<rad
assert np.sum(active&activep)==1
b=svg(320,320,'c9-circle','S ve S asal tam bir dairesel özniteliği ortak etkinleştiriyor')+txt(160,22,'Ortak alıcı alan:1',16)
for j,(cx,cy)in enumerate(centers):b+=f'<circle cx="{30+240*cx}" cy="{280-240*cy}" r="{rad*240}" stroke="var(--ink-3)" fill="var(--c-a)" fill-opacity="{.16 if active[j] else .02}"/>'
for s,t in [(S,'S'),(SP,'S′')]:b+=txt(30+240*s[0],280-240*s[1],'+',20)+txt(38+240*s[0],268-240*s[1],t,14,'start')
b+=txt(160,306,'Koyuluk: S için etkin alanlar',12)
parts=[b+'</svg>'];cent=np.array([(a,b)for a in np.linspace(.05,.95,7)for b in np.linspace(.05,.95,7)])
for j,(rx,ry)in enumerate([(.18,.18),(.35,.35),(.12,.5)]):
 act=np.sum(((cent-[.5,.5])/[rx,ry])**2,axis=1)<1
 F=np.zeros_like(X)
 for c in cent[act]:F+=(((X-c[0])/rx)**2+((Y-c[1])/ry)**2<1)
 parts.append(heat(['Dar daire','Geniş daire','Dikey elips'][j],1-F/max(1,act.sum()),f'c9-coarse{j}',plus=(117,122)))
fig(7,parts,'Şekil9.6 ve9.7: solda S/S′tam1ortak alan; sağdaki üç panelde aynı49merkez/yoğunluk, dar-geniş-uzamış alıcı alanlar. Artı eğitim noktası, koyuluk x(s)ᵀx(S)/||x(S)||². Değişimin büyüklüğü yalnız uzaklık değil ortak alan sayısıdır; alanları union olarak yayılır. Kendi merkezleri/yarıçapları kaynakta verilmemiş geometrinin yerine açıkça seçildi (hesaplandı).')
# Same 10240 actual samples and same 50 centers for all three widths.
rng=np.random.default_rng(930001);samples=rng.uniform(0,2,10240);cs=(np.arange(50)+.5)/25;qx=np.linspace(0,2,201);snap=[10,40,160,640,2560,10240];wave=[];parts=[]
for width in [.2,.4,1.]:
 w=np.zeros(50);records=[]
 for t,s in enumerate(samples,1):
  a=abs(cs-s)<width/2;pred=w[a].sum();w[a]+=.2*((.5<s<1.5)-pred)/max(1,a.sum())
  if t in snap:
   y=(abs(qx[:,None]-cs)<width/2)@w;records.append(y.tolist())
 for t,y in zip(snap,records):
  p=plot(f'genişlik{width:g}; örnek{t}',[dict(name='öğrenilen',x=qx,y=y,color=C[0]),dict(name='kare dalga',x=[0,.5,.5,1.5,1.5,2],y=[0,0,1,1,0,0],color=C[1],dash=True)],'x','tahmin',f'c9-wave{width}-{t}',ylim=(-.3,1.5))
  # A scale bar for the actual feature width, away from curve/legend.
  p=p.replace('</svg>',line(110,353,110+width*90,353,C[2],width=3)+txt(245,357,f'alan {width:g}',11)+'</svg>');parts.append(p)
 wave.append(dict(width=width,snapshots=records))
fig(8,parts,'Şekil9.8:18gerçek öğrenme kesiti; her genişlikte10/40/160/640/2560/10240örnek. Üç genişlik için aynı50sabit merkez ve aynı uniform örnek dizisi; α=.2/aktif sayı. Kaynak kesin eksen/genişlik vermediğinden kendi [0,2] aralığımız ve(.5,1.5) kare dalgamız kullanıldı. Alan genişliği alttaki çizgilerle gösterilir; son çözünürlük genişlikten bağımsız olmak zorunda değildir (simüle edildi).')
parts=[]
for tiling in range(4):
 b=svg(240,266,f'c9-tile{tiling}','Dört döşemenin biri; 4×4 çizilmiş karo ve etkin karo')+txt(120,20,f'Döşeme{tiling+1}:16karo',14);off=np.array([1,3])*tiling/4;point=np.array([1.2,.8]);cell=np.floor(point+off).astype(int)
 for r in range(4):
  for c in range(4):
   xx=24+(c-off[0])*44;yy=210-(r-off[1]+1)*44
   # Use each grid's local display box, label common point in that display.
   xx=24+c*44;yy=34+(3-r)*44;isactive=(c,r)==tuple(cell)
   b+=f'<rect x="{xx}" y="{yy}" width="44" height="44" fill="{C[tiling]if isactive else "var(--paper)"}" fill-opacity="{.35 if isactive else 1}" stroke="var(--ink-3)"/>'
 pos=point+off;b+=txt(24+44*pos[0],34+44*(4-pos[1]),'+',18)+txt(120,236,f'ofset({off[0]:g},{off[1]:g})',12)+txt(120,254,'ortak iç nokta; bir aktif karo',11);parts.append(b+'</svg>')
fig(9,parts,'Şekil9.9:4adet4×4çizilmiş ızgara=64gösterilen özellik, aynı iç nokta için4aktif karo. Her panel noktanın o döşemenin koordinatlarındaki konumunu gösterir. Döşemeler kesirli karo genişliğiyle kayar; çizilen64karo bütün kaydırılmış sınırlı alanın kenar örtüşmesi iddiası değildir. Genel uygulama gerekli dış kenar karolarını da içerir (hesaplandı).')
fig(10,[plot('Tek döşeme / 50 döşeme',[dict(name=c['name'],x=base['checkpoints'],y=c['results']['mean'],color=C[j])for j,c in enumerate(base['curves'][6:])],'bölüm','kök µ-VE','c9-tiles',ylim=(0,.5))],'Şekil9.10: genişlik200durum; tek döşemeα.0001,50döşemeα.0001/50 ve ofsetler4durum.30koşu×5000tam bölümde son kökVE .1136452 ve .0703186. İlk noktadaki toplam adım aynı; ince temsille daha düşük hata bu görev/tohum/parametrelere aittir (simüle edildi).')
def ids(x,y,asym):return [(i,math.floor(x+i/8),math.floor(y+(3 if asym else 1)*i/8))for i in range(8)]
parts=[]
for asym in [False,True]:
 b=svg(220,252,'c9-stencil'+str(asym),'Üç renkli döşeme ofsetlerinin hesaplanan şeması')+txt(110,20,'(1,3) ofset'if asym else'(1,1) ofset',15)
 for i in range(3):
  dx=i*12;dy=i*(3 if asym else 1)*12;b+=f'<rect x="{35+dx}" y="{40+dy}" width="110" height="110" fill="none" stroke="{C[i]}" stroke-width="2"/>'
 b+=txt(110,238,'adım: karo genişliği / 8',12);parts.append(b+'</svg>')
 for k in range(8):
  sx=(k+.5)/8;sy=.0625;A=set(ids(sx,sy,asym));coords=np.linspace(-1,1,80);F=np.array([[sum(i in A for i in ids(sx+dx,sy+dy,asym))/8 for dx in coords]for dy in coords[::-1]])
  parts.append(heat(('Asimetrik'if asym else'Uniform')+f' örüntü{k+1}',1-F,f'c9-kern{asym}-{k}',axis=False,plus=(117,122)))
fig(11,parts,'Şekil9.11: üstte(1,1), altta(1,3) ofsetli8döşeme için sekizer farklı eğitim fazı; artı öğrenilen durum, gri değer ortak aktif karo sayısı/8. Yan şemalarda ilk3döşeme renkle ayrılır. Eksenler eğitim noktasının±1karo çevresidir. Uniform ofsetin diagonal yönlenmesi ve asimetrik ofsetin daha merkezli örüntüsü gerçek örtüşmelerden çıkar; tam dönel eşdeğerlik değildir (hesaplandı).')
parts=[]
sites=np.random.default_rng(930002).uniform(-.1,1.1,(28,2));vor=Voronoi(sites)
for kind in range(3):
 b=svg(230,260,f'c9-part{kind}','Düzensiz, logaritmik ve diyagonal uzay bölmeleri')+txt(115,20,['Düzensiz çokgen','Logaritmik dik şerit','Diyagonal şerit'][kind],14)
 b+=f'<defs><clipPath id="c9-partclip{kind}"><rect x="25" y="40" width="180" height="180"/></clipPath></defs><g clip-path="url(#c9-partclip{kind})">'
 if kind==0:
  for (site1,site2),(a,c) in zip(vor.ridge_points,vor.ridge_vertices):
   if a>=0 and c>=0:p,q=vor.vertices[[a,c]]
   else:
    p=vor.vertices[max(a,c)];tangent=sites[site2]-sites[site1];tangent/=np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]]);midpoint=(sites[site1]+sites[site2])/2
    direction=np.sign(np.dot(midpoint-sites.mean(axis=0),normal))*normal;q=p+100*direction
   b+=line(25+180*p[0],220-180*p[1],25+180*q[0],220-180*q[1])
 elif kind==1:
  for v in np.r_[0,np.geomspace(.025,1,13)]:b+=line(25+180*v,40,25+180*v,220)
 else:
  for a in np.arange(-1,2.01,.2):b+=line(25+180*a,40,25+180*(a+1),220)
 b+='</g><rect x="25" y="40" width="180" height="180" fill="none" stroke="var(--ink)"/>'+txt(115,246,['Voronoi:28sabit merkez','Sol yönde daha ince ayrım','Şerit boyunca genelleme'][kind],12);parts.append(b+'</svg>')
fig(12,parts,'Şekil9.12: üç ayrık partition kendi koordinatlarımızla üretildi:Voronoi çokgenleri, solda yoğun logaritmik dik şeritler, diagonal şeritler. Uzayan yön değerleri geneller; ince kalan yön daha iyi ayırır. Yalnız yatay+dikey şeritler bütün konjonksiyonları temsil etmez; dikdörtgenler de gerekebilir (hesaplandı).')
def hashint(v):
 v=((v^(v>>16))*0x7feb352d)&0xffffffff;v=((v^(v>>15))*0x846ca68b)&0xffffffff;return(v^(v>>16))%16
buckets={j:[]for j in range(16)}
for r in range(8):
 for c in range(8):buckets[hashint(8*r+c)].append((r,c))
bucket=next(j for j,v in buckets.items()if len(v)==4 and all(max(abs(a-c),abs(b-d))>1 for k,(a,b)in enumerate(v)for c,d in v[k+1:]));cells=buckets[bucket]
b=svg(360,310,'c9-hash','Aynı hash kovasına düşen dört ayrık bölge')+txt(180,20,f'Tutarlı hash → kova{bucket}',16)
for r in range(8):
 for c in range(8):b+=f'<rect x="{20+c*24}" y="{40+r*24}" width="24" height="24" fill="{C[0]if(r,c)in cells else "var(--paper)"}" fill-opacity="{.45 if(r,c)in cells else 1}" stroke="var(--ink-3)" stroke-width=".6"/>'
b+=box(263,112,70,48,'tek özellik',color='var(--paper)')
for r,c in cells:b+=line(32+c*24,52+r*24,260,136,C[0],arrow='c9-hash')
b+=txt(180,272,'64bölge →16kova; seçili kovada4bölge',12)+txt(180,293,'Ayrık yerler aynı ağırlığı paylaşabilir',12)
fig(13,[b+'</svg>'],'Kaynak s.221numarasız şekil: dört ayrık kare aynı hash özelliğine bağlandı. Kendi deterministik32bit karma fonksiyonumuzun64hücre/16kova eşlemesinden gerçekten4hücre içeren kova seçildi. Hash bellek sınırını kontrol eder; ilgisiz yerler arası genelleme ve çakışma öğrenme hızını değiştirebilir (hesaplandı).')
xx=np.linspace(-2.5,2.5,301);p=plot('Gauss RBF: komşu üç merkez',[dict(name=f'c={c}',x=xx,y=np.exp(-.5*((xx-c)/.45)**2),color=C[i])for i,c in enumerate([-1,0,1])],'s','x_i(s)','c9-rbf',ylim=(0,1.1))
p=p.replace('</svg>',line(181.5,54,181.5,245,'var(--ink-3)',True)+line(181.5,166,205.9,166,C[1],width=3)+txt(214,161,'σ=.45',11,'start')+'</svg>')
fig(14,[p],'Şekil9.13: üç merkez cᵢ₋₁=−1,cᵢ=0,cᵢ₊₁=1; σ=.45. Orta merkez kesik dikey,σ yatay çizgiyle gösterildi. x=exp(−(s−c)²/(2σ²)) hesaplandı; merkezler/genişlikler sabitken öğrenilen çıktı ağırlıklarda doğrusaldır (hesaplandı).')
b=svg(360,340,'c9-ann','4 girdi, iki dört birimli gizli katman ve iki çıktı; 40 hesaplanmış bağlantı')+txt(180,22,'4 →4 →4 →2;40bağlantı',17);layers=[[(35,65+50*i)for i in range(4)],[(130,65+50*i)for i in range(4)],[(225,65+50*i)for i in range(4)],[(320,115+50*i)for i in range(2)]]
for left,right in zip(layers,layers[1:]):
 for x,y in left:
  for xx,yy in right:b+=line(x+9,y,xx-9,yy,'var(--ink-3)',width=.8)
for j,ly in enumerate(layers):
 for x,y in ly:
  b+=f'<circle cx="{x}" cy="{y}" r="9" fill="var(--paper)" stroke="{C[j]}"/>'
  if j==0:b+=line(8,y,x-11,y,arrow='c9-ann')
  if j==3:b+=line(x+11,y,353,y,arrow='c9-ann')
 b+=txt(ly[0][0],256,['girdi','gizli1','gizli2','çıktı'][j],12)
b+=txt(180,293,'Dış değerler →aktivasyonlar →çıktılar',12)+txt(180,316,'Biaslar çizilmedi;10bias ile50parametre',12)
fig(15,[b+'</svg>'],'Şekil9.14: iki gizli katmanlı ileri beslemeli ağ. Kenarlar4×4+4×4+4×2=40 olarak hesaplandı; giriş ve çıkış okları korundu. Birimlerin doğrusal olmayan aktivasyonları anlamlı derinlik sağlar; tümü afin olursa bileşke yine afin olur (hesaplandı).')
# Synthetic A and actual valid 5x5 convolution and 2x2 average pooling.
yy,xx=np.mgrid[:32,:32];A=(((abs(xx-(16-(yy-5)*.38))<1.7)|(abs(xx-(16+(yy-5)*.38))<1.7))&(yy>=5)&(yy<=27))|((yy>=18)&(yy<=20)&(xx>=10)&(xx<=22));A=A.astype(float)
kernel=np.tile(np.array([-1,-1,0,1,1])[:,None],(1,5))/10
C1=np.array([[np.sum(A[r:r+5,c:c+5]*kernel)for c in range(28)]for r in range(28)]);S2=C1.reshape(14,2,14,2).mean(axis=(1,3))
fig(16,[heat('Sentetik A:32×32',1-A,'c9-A',False),heat('C1örneği:28×28',(C1+1)/2,'c9-C1',False),heat('S2ortalama:14×14',(S2+1)/2,'c9-S2',False)],'Şekil9.15 ile bağlantılı kendi görüntü hesabı: sentetik A üzerinde bir5×5yatay kenar çekirdeği stride1 ile uygulanıp2×2ortalama alındı. C1komşu alıcı alanları4satır/4sütun paylaşır, S2alanları çakışmaz. Özgün LeNet6C1haritasının tamamı ve32²→6×28²→6×14²→16×10²→16×5²→120→84→10Gauss çıkış mimarisi M4.2 /LeNet bölümündeki mevcut kendi şemada gösterilir. Bu paneller öğrenilmiş LeNet ağırlıkları değildir (hesaplandı).')
b=svg(350,260,'c9-interest','Dört durumlu ilgi zinciri: gerçek değerler4,3,2,1, paylaşılan ağırlıklarw1,w1,w2,w2')+txt(175,23,'İlgi ve vurgu: Örnek9.4',16)
for j in range(4):
 x=35+j*78;b+=f'<circle cx="{x}" cy="95" r="19" fill="var(--paper)" stroke="{C[j//2]}"/>'+txt(x,100,'w'+str(1+j//2),14)+txt(x,55,'I='+str(int(j==0)),13)+txt(x,145,'v='+str(4-j),14)
 if j<3:b+=line(x+21,95,x+55,95,arrow='c9-interest')+txt(x+39,83,'+1',12)
b+=line(289,95,312,95,arrow='c9-interest')+txt(302,81,'+1',12)+f'<rect x="317" y="85" width="20" height="20" fill="var(--paper)" stroke="var(--ink)"/>'+txt(175,185,'Başlangıç solda; γ1; terminal kare',13)+txt(175,214,'n2 → M=(1,0,1,0)',14)+txt(175,243,'MC → M=I=(1,0,0,0)',14)
fig(17,[b+'</svg>'],'Kaynak s.235numarasız zincir: dört duruma ilgi(1,0,0,0), gerçek değerler(4,3,2,1), paylaşılan ağırlıklar(w1,w1,w2,w2), her geçiş+1 ve terminal kare. İki adımlı vurgu başlangıç ilgisini bootstrap yapılan üçüncü duruma taşır. Düz MC/2-adım limit(3.5,1.5), ilgi MC(4,0), vurgulu2-adım(4,2); azalan adım koşuluyla (hesaplandı).')
metrics=dict(wave=dict(seed=930001,centers=cs.tolist(),domain=[0,2],samples=10240,snapshots=snap,curves=wave),coarse=dict(centers=centers.tolist(),S=S.tolist(),S_prime=SP.tolist(),radius=rad,shared=int(np.sum(active&activep))),hash=dict(bucket=bucket,cells=cells),syntheticCNN=dict(input_sum=float(A.sum()),C1_shape=list(C1.shape),S2_shape=list(S2.shape),C1_sum=float(C1.sum()),S2_sum=float(S2.sum())))
(BASE/'measurements_ch9_figures.json').write_text(json.dumps(metrics,indent=2)+'\n')
data_dir=BASE.parent.parent/'site/js/data'
data_dir.mkdir(parents=True,exist_ok=True)
for name,data in [('walk',D),('exact',{k:E[k]for k in ['V','mu','reward','groups','expected_episode_length']})]:
 (data_dir/f'ch9-{name}.js').write_text('// Own computed data; reproducible scripts in notes/sutton-2018.\nexport default '+json.dumps(data,separators=(',',':'))+';\n')
print('16 new figure groups; 18 square-wave curves; 16 tile kernels; own computed geometry.')

