"""Own computed chapter11 figures. python generate_ch11.py OUT; no source pixels/curves."""
from pathlib import Path
import sys,json,math,html,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent;OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
E=json.loads((BASE/'measurements_ch11_reference.json').read_text());S=json.loads((BASE/'measurements_ch11_samples.json').read_text())
C=['var(--c-a)','var(--c-b)','var(--c-p)','var(--c-q)','var(--ink)'];sub=str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
def text(x,y,s,size=12,anchor='middle',color='var(--ink)'):
 return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color}">{html.escape(str(s))}</text>'
def start(w,h,aria,key):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(aria,quote=True)}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto"><defs><marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def line(x,y,u,v,c='var(--ink)',dash=False,arrow=None,width=1.3):return f'<path d="M{x:.3f},{y:.3f} L{u:.3f},{v:.3f}" fill="none" stroke="{c}" stroke-width="{width}"'+(' stroke-dasharray="4 3"'if dash else'')+(f' marker-end="url(#{arrow})"'if arrow else'')+'/>'
def path(points,c='var(--ink)',dash=False,width=1.3):return '<path d="'+' '.join(('M'if i==0 else'L')+f'{x:.3f},{y:.3f}'for i,(x,y)in enumerate(points))+'" fill="none" stroke="'+c+'" stroke-width="'+str(width)+'"'+(' stroke-dasharray="4 3"'if dash else'')+'/>'
def fig(n,parts,cap):
 (OUT/f'f{n}.html').write_text('<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">'+''.join(parts)+'</div><figcaption>Şekil '+str(n)+' — '+cap+'</figcaption></figure>')
def plot(title,series,xlabel,ylabel,ylim=None,xlim=None,log=False,ticks=None):
 key='c11-'+hashlib.sha256((title+xlabel+ylabel).encode()).hexdigest()[:12];L,R,T,B=48,314,50,235;w,h=330,350
 xs=[x for s in series for x in s['x']];ys=[v for s in series for v in s['y']if v is not None and math.isfinite(v)];lo,hi=xlim or(min(xs),max(xs));yl,yh=ylim or(min(0,min(ys)),max(ys)*1.06)
 if log:
  pos=[v for v in ys if v>0];yl,yh=ylim or(10**math.floor(math.log10(min(pos))),10**math.ceil(math.log10(max(pos))))
 if yh==yl:yh=yl+1
 X=lambda x:L+(R-L)*(x-lo)/(hi-lo or 1)
 Y=lambda y:B-(B-T)*(math.log10(max(y,yl)/yl)/math.log10(yh/yl)if log else(y-yl)/(yh-yl))
 b=start(w,h,title,key)+text(165,22,title,14)+text(L,42,ylabel,11,'start')+line(L,T,L,B)+line(L,B,R,B)
 yt=ticks or([10**i for i in range(math.floor(math.log10(yl)),math.ceil(math.log10(yh))+1)]if log else[yl+(yh-yl)*i/4 for i in range(5)])
 for y in yt:
  if not yl<=y<=yh:continue
  label=('10'+str(int(round(math.log10(y)))).translate(str.maketrans('0123456789-','⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))if log else f'{y:.3g}')
  b+=line(L,Y(y),R,Y(y),'var(--ink-3)',True)+text(L-4,Y(y)+4,label,10,'end')
 for x in[lo,(lo+hi)/2,hi]:b+=text(X(x),252,f'{x:.4g}',10,'end'if x==hi else'middle')
 b+=f'<defs><clipPath id="{key}-clip"><rect x="{L}" y="{T}" width="{R-L}" height="{B-T}"/></clipPath></defs>'
 b+='<g clip-path="url(#'+key+'-clip)">'
 for i,s in enumerate(series):
  pts=[(X(x),Y(y))for x,y in zip(s['x'],s['y'])if y is not None and math.isfinite(y)and(not log or y>0)];b+=path(pts,s.get('color',C[i%5]),s.get('dash',False),1.3)
 b+='</g>'+text((L+R)/2,274,xlabel,11)
 labels=[]
 for i,s in enumerate(series):
  if s['name']and s['name']not in[x[0]for x in labels]:labels.append((s['name'],s.get('color',C[i%5]),s.get('dash',False)))
 for i,(name,c,dash)in enumerate(labels):b+=line(12,291+13*i,31,291+13*i,c,dash)+text(38,295+13*i,name,10,'start')
 return b+'</svg>'

def diagram(title,layers,edges,labels,key,term=(),alias=None,foot=()):
 w,h=330,320;b=start(w,h,title,key)+text(165,22,title,14);pos={}
 for col,names in enumerate(layers):
  for row,name in enumerate(names):pos[name]=((165 if len(layers)==1 else 45+240*col/(len(layers)-1)),85+135*(row+.5)/len(names))
 if alias:
  xs=[pos[z][0]for z in alias];ys=[pos[z][1]for z in alias];b+=f'<rect x="{min(xs)-27}" y="{min(ys)-32}" width="{max(xs)-min(xs)+54}" height="{max(ys)-min(ys)+64}" rx="24" fill="var(--c-p)" fill-opacity=".08" stroke="var(--c-p)" stroke-dasharray="4 3"/>'
 pairs={(a,z)for a,z,_ in edges}
 for a,z,label in edges:
  x,y=pos[a];u,v=pos[z]
  if a==z:
   crowded=any(n!=a and abs(px-x)<1 and py<y for n,(px,py)in pos.items())
   if crowded:d=f'M{x+17},{y-13} C{x+40},{y-40} {x+40},{y+40} {x+17},{y+13}';lx,ly=x,y+49
   else:d=f'M{x-13},{y-17} C{x-50},{y-75} {x+50},{y-75} {x+13},{y-17}';lx,ly=x,y-60
   b+=f'<path d="{d}" fill="none" stroke="var(--ink)" marker-end="url(#{key})"/>'+text(lx,ly,label,10)
  else:
   dx,dy=u-x,v-y;dist=math.hypot(dx,dy);ux,uy=dx/dist,dy/dist;r1=20;r2=8 if z in term else 20;sx,sy=x+r1*ux,y+r1*uy;ex,ey=u-r2*ux,v-r2*uy
   if(z,a)in pairs:
    cx,cy=(sx+ex)/2-uy*22,(sy+ey)/2+ux*22;d=f'M{sx},{sy} Q{cx},{cy} {ex},{ey}';b+=f'<path d="{d}" fill="none" stroke="var(--ink)" marker-end="url(#{key})"/>';lx,ly=.25*sx+.5*cx+.25*ex,.25*sy+.5*cy+.25*ey
   else:b+=line(sx,sy,ex,ey,arrow=key);lx,ly=(sx+ex)/2,(sy+ey)/2
   b+=text(lx+14 if abs(dy)>abs(dx)else lx,ly+4 if abs(dy)>abs(dx)else ly-6,label,10)
 for name,(x,y)in pos.items():
  if name in term:b+=f'<rect x="{x-7}" y="{y-7}" width="14" height="14" fill="var(--ink-3)" stroke="var(--ink)"/>'
  else:b+=f'<circle cx="{x}" cy="{y}" r="20" fill="var(--paper)" stroke="var(--ink)"/>'+text(x,y+4,labels.get(name,name),12)
 for i,t in enumerate(foot):b+=text(165,264+i*17,t,11)
 return b+'</svg>'

# Fragment from the exact scalar update, not a complete MDP.
parts=[]
for t in range(3):
 w=10*(1+.1*(2*.99-1))**t
 parts.append(diagram(f'Güncelleme {t}',[['A'],['B']],[('A','B','R=0')],{'A':f'{w:.5g}','B':f'{2*w:.5g}'},f'c11-frag{t}',foot=('v(A)=w, v(B)=2w','α=0.1, γ=0.99')))
fig(22,parts,'Kaynak s.260 numarasız w→2w parçası: özellikler1/2, başlangıç w10. Aynı geçiş tek başına yinelenince w çarpanı1+α(2γ−1). γ0.99 ile10→10.98→12.05604; kaynak γ≈1 için yaklaşık11/12.1 der. Tek eylemdeρ1; örnek tam bir MDP değildir. Off-policy diğer geçişleriρ0 ile devre dışı bırakabilir (hesaplandı).')
# Baird: bundling dashed transitions avoids42 overlapping arrows but preserves the kernel.
key='c11-baird-network';b=start(330,345,'Baird yedi durum, sekiz özellik ve iki eylem',key)+text(165,21,'Baird: R=0, γ=0.99',14);top=[(30+54*i,100)for i in range(6)];bottom=(165,218)
for i,(x,y)in enumerate(top):b+=line(x,y+20,165,198,arrow=key)+line(x,80,x,49,dash=True,arrow=key)+line(x,49,x,80,dash=True,arrow=key)
b+=line(30,46,300,46,dash=True)+path([(147,218),(12,218),(12,46),(30,46)],dash=True)
b+=f'<path d="M181,229 C246,288 258,183 184,208" fill="none" stroke="var(--ink)" marker-end="url(#{key})"/>'
for i,(x,y)in enumerate(top):b+=f'<circle cx="{x}" cy="{y}" r="20" fill="var(--paper)" stroke="var(--ink)"/>'+text(x,y+3,'2w'+str(i+1).translate(sub)+'+w₈',10)
b+=f'<circle cx="165" cy="218" r="20" fill="var(--paper)" stroke="var(--ink)"/>'+text(165,222,'w₇+2w₈',10)
for i,t in enumerate(['Kesik: altı üst duruma eşit olasılık','b(kesik)=6/7; b(düz)=1/7','π(düz)=1; düz her yerden alt duruma']):b+=text(165,286+i*17,t,11)
fig(23,[b+'</svg>'],'Kaynak Şekil11.1 modelinden hesaplanan ağ. Kesik üst hat, her başlangıçtan altı üst duruma1/6 geçişi birleştirerek gösterir. Davranış durum dağılımı düzgün7; hedef uzun dönemde alt durumda. Sekiz ağırlık/yedi bağımsız durum özelliği: w0 çözümü var ama tek çözüm değil. Kaynaktaki episodik sözcüğüne rağmen bu şemada terminal yok; hesaplarımız açıkça devam eden yedi durumlu modeldir (modelden türetildi).')

def weight_series(rows,key='w'):
 return [dict(name='w₁..w₆'if i==0 else''if i<6 else'w₇'if i==6 else'w₈',x=[z['t']for z in rows],y=[z[key][i]for z in rows],color=C[0 if i<6 else 2 if i==6 else 1],dash=i<6 and i%2==1)for i in range(8)]
case=S['samples'][0];sample=case['sample_history'];rows=[dict(t=z[0],w=None)for z in sample]
# Reconstruct source sample weights through the independently replayed stream for plotting every5steps.
# C++ output stores final values/RMS; exact deterministic point weights are generated separately below.
class RNG:
 def __init__(self,s):self.s=s
 def next(self):
  self.s=(self.s+0x6d2b79f5)&0xffffffff;t=self.s;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296
X=np.array(E['baird']['X']);rng=RNG(1);state=int(rng.next()*7);Ws=[np.array(E['baird']['w0'])for _ in range(2)];u=np.zeros(8);hist=[[],[]]
for t in range(1001):
 if t%5==0:
  for k in range(2):hist[k].append(dict(t=t,w=Ws[k].tolist()))
 if t==1000:break
 solid=rng.next()>=6/7;ns=6 if solid else int(rng.next()*6);rho=7 if solid else 0;x,xp=X[state],X[ns]
 delta=.99*sum(Ws[0]*xp)-sum(Ws[0]*x);Ws[0]+=.01*rho*delta*x
 delta=.99*sum(Ws[1]*xp)-sum(Ws[1]*x);vx=sum(u*x);dw=.005*rho*(delta*x-.99*xp*vx);u+=.05*rho*(delta-vx)*x;Ws[1]+=dw;state=ns
for k,idx in[(0,0),(1,1)]:assert max(abs(Ws[k]-np.array(case['final'][idx]['weights'])))<1e-9
fig(24,[plot('SG-TD: gerçek örnek, tohum1',weight_series(hist[0]),'gerçek geçiş','ağırlık'),plot('SG-DP: tam beklenen güncelleme',weight_series(E['baird']['profiles']['DP']['history']),'süpürme','ağırlık')],'Kaynak Şekil11.2 karşılığı:1000 örnek geçiş ve1000 tam süpürme; α0.01, γ0.99, w başlangıcı(1,1,1,1,1,1,10,1). Beklenen güncellemede her durumun ağırlığı1/7. Tohum1 örnek norm335.79573, DP norm436.85602. Gürültüyü kaldırmak bu örnekte kararsızlığı gidermez. Renkler grupları ayırır; üst altı örnek eğri ayrı ayrı çizildi (simüle edildi/hesaplandı).')
eps=.05;g=.99;times=list(range(101));off=g*(6-4*eps)/5;on=g*(4-2*eps)/(4+eps)
fig(25,[diagram('Tsitsiklis–Van Roy modeli',[['A'],['B'],['T']],[('A','B','0'),('B','B','0; 1−ε'),('B','T','0; ε')],{'A':'w','B':'2w'},'c11-tvr',term=['T'],foot=('ε=0.05; γ=0.99 (kendi seçimimiz)','Bütün ödüller0; hedef değer0')),plot('Her adımda tam LS uydurma',[dict(name='iki duruma eşit ağırlık',x=times,y=[10*off**t for t in times]),dict(name='episodik ziyaret ağırlığı',x=times,y=[10*on**t for t in times])],'LS yinelemesi','|w| (log)',log=True)],'Kaynak Örnek11.1 numarasız şema: w yeni=γ(6−4ε)w/5. Seçtiğimiz ε0.05,γ0.99 için çarpan1.1484; tam LS bile ıraksar. Episodik µ=(ε,1)/(1+ε) ile çarpan γ(4−2ε)/(4+ε)=0.9533333. Fark yaklaşık çözümün adım büyüklüğü değil, projeksiyon dağılımıdır (hesaplandı).')
# The real weighted3D geometry: D^1/2 maps the metric to ordinary Euclidean distance.
geom=E['geometry'];Z=np.array(geom['X']);mu=np.array(geom['mu']);rootD=np.diag(np.sqrt(mu));Q,_=np.linalg.qr(rootD@Z,mode='complete');true=np.array(geom['true_value'])
if (Q.T@rootD@true)[2]<0:Q[:,2]*=-1
coord=lambda v:Q.T@rootD@np.asarray(v)
raw=lambda v:np.array([v[0]-.55*v[1],-.25*v[0]-.35*v[1]-v[2]])
v0=np.zeros(3);bv=np.array(geom['r']);pbv=np.array(geom['Pi'])@bv
allv=[v0,bv,pbv,true]+[Z@np.array(z['w'])for z in geom['metrics'].values()]+[np.array(z)for z in geom['Bellman_iterates']+geom['projected_iterates']]
corners=[Z@np.array(z)for z in[[-2.2,-2.2],[.6,-2.2],[.6,.6],[-2.2,.6]]];allv+=corners+[Z[:,0]*.6,Z[:,1]*.6]
xy=np.array([raw(coord(z))for z in allv]);mins=xy.min(0)-.35;maxs=xy.max(0)+.35;sx=270/(maxs[0]-mins[0]);sy=205/(maxs[1]-mins[1]);project=lambda z:(35+(raw(coord(z))[0]-mins[0])*sx,50+(raw(coord(z))[1]-mins[1])*sy)
key='c11-geometry';b=start(330,350,'Hesaplanmış ağırlıklı değer uzayı ve Bellman projeksiyonu',key)+text(165,20,'√µ ile ölçeklenmiş değer uzayı',13)
b+='<polygon points="'+' '.join(f'{project(z)[0]:.3f},{project(z)[1]:.3f}'for z in corners)+'" fill="var(--c-a)" fill-opacity=".06" stroke="var(--ink-3)"/>'
for trace in['Bellman_iterates','projected_iterates']:
 pts=[project(z)for z in geom[trace]];b+=path(pts,'var(--ink-3)',False,1)
for a,z,label,c in[(v0,bv,'BE',C[1]),(v0,pbv,'PBE',C[0]),(np.array(geom['Pi'])@true,true,'VE',C[2])]:
 x,y=project(a);u,v=project(z);b+=line(x,y,u,v,c,arrow=key,width=2)+text((x+u)/2+(-9 if label=='PBE'else 9),(y+v)/2,label,11,'end'if label=='PBE'else'start',c)
x,y=project(bv);u,v=project(pbv);b+=line(x,y,u,v,'var(--ink-3)',True)
for value,label,dx,dy in[(v0,'v_w (w=0)',-60,17),(bv,'Bπv_w',7,-7),(pbv,'ΠBπv_w',9,14),(true,'vπ',5,-8)]:
 x,y=project(value);b+=f'<circle cx="{x}" cy="{y}" r="3" fill="var(--ink)"/>'+text(x+dx,y+dy,label,10,'start')
for value,label,dx,dy in[(np.array(geom['Pi'])@true,'Πvπ',-32,15),(Z@np.array(geom['metrics']['PBE']['w']),'wTD',7,17)]:
 x,y=project(value);b+=f'<circle cx="{x}" cy="{y}" r="3" fill="var(--ink)"/>'+text(x+dx,y+dy,label,10,'start')
for i in range(2):
 end=Z[:,i]*.6;x,y=project(v0);u,v=project(end);b+=line(x,y,u,v,'var(--ink-3)',arrow=key)+text(u,v+13,'w'+str(i+1),10)
b+=text(165,292,'Gri: tam Bellman / projeksiyonlu yol',10)+text(165,310,'Düzlem: temsil edilebilir Xw değerleri',10)+text(165,328,'Ağırlıklı izdüşüm çizgileri hesaplandı',10)
# Parameter-plane minima with actual coordinates, no arbitrary point placement.
b2=start(330,350,'Dört amacın hesaplanmış farklı minimumları','c11-minima')+text(165,20,'Parametre düzlemindeki minimumlar',13);L,R,T,B=50,305,50,260;lo,hi=-2.,.1
XX=lambda v:L+(v-lo)/(hi-lo)*(R-L);YY=lambda v:B-(v-lo)/(hi-lo)*(B-T)
b2+=line(L,T,L,B)+line(L,B,R,B)+text(178,283,'w1',12)+text(26,150,'w2',12)
offsets={'VE':(6,-10),'BE':(5,15),'PBE':(-5,18),'TDE':(8,0)}
for k,(name,z)in enumerate(geom['metrics'].items()):
 x,y=XX(z['w'][0]),YY(z['w'][1]);dx,dy=offsets[name];b2+=f'<circle cx="{x}" cy="{y}" r="4" fill="{C[k]}"/>'+text(x+dx,y+dy,'min '+name,11,'start'if dx>=0 else'end',C[k])
for z in[-2,-1,0]:b2+=text(XX(z),277,z,10)+text(43,YY(z)+4,z,10,'end')
b2+=text(165,312,'γ0.9; µ gerçek durağan dağılım',10)+text(165,329,'min TDE='+f'{geom["metrics"]["TDE"]["TDE"]:.6f}',10)
fig(26,[b+'</svg>',b2+'</svg>'],'Kaynak Şekil11.3 karşılığı kendi üç durumlu modelimizde tümüyle hesaplandı. P/r/X etkileşimli geometri deneyinde açık; µ politikanın durağan dağılımı. QR, √µ ile ölçeklenmiş üç boyutlu uzayda temsil düzlemini kurar. Tam Bellman yolu vπ, projeksiyonlu yol wTD sabit noktasına gider. VE/BE/PBE/TDE minimumları farklı; kaynak çizimindeki TDE=0 etiketi genel stokastik durumda geçerli değildir, burada minimum TDE0.34270625. Minimum VE0.36717624,BE0.033226965,PBE sayısal0. Noktalar ve oklar modelden hesaplandı.')
# Source A-split and A-presplit, with expected-error quantities from exact models.
split=E['A_split'];presplit=E['A_presplit']
fig(27,[diagram('A-split: üç durum',[['A'],['B','C'],['T1','T2']],[('A','B','0; 1/2'),('A','C','0; 1/2'),('B','T1','1'),('C','T2','0')],{},'c11-split',term=['T1','T2'],foot=('γ1; her bölüm A ile başlar','v(A)=1/2; v(B)=1; v(C)=0','Terminal iki yerde gösterildi')),plot('A-split: iki tahminin değerleri',[dict(name='doğru değerler',x=[0,1,2],y=[.5,1,0]),dict(name='TDE minimumu',x=[0,1,2],y=split['metrics']['TDE']['w'])],'gözlenen A / B / C','değer',(0,1.1))],'Kaynak Örnek11.2 numarasız diyagramı ve hesaplanmış çözümler. Doğru değerlerde iki adımın TDE ortalaması1/8; naif residual minimumunda (1/2,3/4,1/4), her adımın kare hatası1/16. Daha düşük TDE doğru değer demek değildir; BE minimumu A-split içinde doğru değerleri verir (hesaplandı).')
fig(28,[diagram('A-presplit: dört gerçek durum',[['A1','A2'],['B','C'],['T1','T2']],[('A1','B','0'),('A2','C','0'),('B','T1','1'),('C','T2','0')],{'A1':'A₁','A2':'A₂'},'c11-presplit',term=['T1','T2'],alias=['A1','A2'],foot=('A₁/A₂ başlangıçları1/2; φ(A₁)=φ(A₂)','Üç gözlem: A/B/C; γ1','Sayılar terminal dışı durumlar içindir')),plot('Aynı gözlenen veri, farklı BE',[dict(name='A-split: BE',x=[0,1],y=[split['metrics']['BE']['BE'],split['metrics']['TDE']['BE']]),dict(name='A-presplit: BE',x=[0,1],y=[presplit['metrics']['VE']['BE'],presplit['metrics']['BE']['BE']])],'0: VE en iyi / 1: naif residual','BE',(0,.14))],'Kaynak Örnek11.3: dört alttaki durum, üç özellik gözlemi. Kitap üç-state derken A1/A2 fiziksel ayrımını şemada gösterir. Beklenen gözlenen akış A-split ile aynıdır. Aynı alttaki duruma çift örnek alabilen oracle, burada deterministik geçişler nedeniyle TDE ile aynı BE minimumunu bulur: A1/A2 ortak1/2,B3/4,C1/4. En iyi ortak değerler(1/2,1,0) için BE1/8; BE minimumu1/16 (hesaplandı).')
fig(29,[diagram('Tek durum: ödül rasgele',[['U']],[('U','U','0 veya2; 1/2')],{'U':'w'},'c11-iid1',foot=('φ=1; γ0 için gerçek değer1','w1 için VE0; RE1')),diagram('İki gizli durum, aynı özellik',[['L'],['R']],[('L','L','0; 1/2'),('L','R','0; 1/2'),('R','L','2; 1/2'),('R','R','2; 1/2')],{'L':'w','R':'w'},'c11-iid2',foot=('Başlangıç1/2; iki durumda da φ=1','γ0: gerçek değerler0/2','w1 için VE1; RE1'))],'Kaynak s.274 numarasız MRP çifti: gözlenen iid0/2 ödül yasası aynıdır. Sağda ödül gerçek duruma koşullanınca deterministiktir; geçiş matrisi yine rasgeledir. VE değeri0 veya1 olabilir, bu yüzden yalnız veriden belirlenmez. RE ve ortak argmin w1 ise belirlenir. Buradaki öğrenilebilirlik, sonsuz veriden ayırt edilebilirliktir; PAC örnek karmaşıklığı tanımı değildir (hesaplandı).')
left=diagram('BE karşı örneği: iki durum',[['A'],['B']],[('A','B','0'),('B','A','+1; 1/2'),('B','B','−1; 1/2')],{},'c11-be-left',foot=('µ(A)=1/3; µ(B)=2/3','w0 bütün γ<1 için BE0'))
right=diagram('Üç durum; B/B′ aynı gözlem',[['A'],['B','Bprime']],[('A','B','0; 1/2'),('A','Bprime','0; 1/2'),('B','A','+1'),('Bprime','B','−1; 1/2'),('Bprime','Bprime','−1; 1/2')],{'Bprime':'B′'},'c11-be-right',alias=['B','Bprime'],foot=('Üç alttaki durumda µ1/3','w0 için BE2/3'))
xs=[i/1000 for i in range(1000)];rawA=[-g/(1+g+g*g)for g in xs];rawB=[g/(2*(1+g+g*g))for g in xs]
fig(30,[left,right,plot('İkinci MRP için minimum parametre',[dict(name='ham wA →−1/3',x=xs,y=rawA),dict(name='ham wB →+1/6',x=xs,y=rawB),dict(name='B0 merkezli A →−1/2',x=xs,y=[a-b for a,b in zip(rawA,rawB)],dash=True)],'γ','parametre',(-.55,.2))],'Kaynak Örnek11.4 numarasız model çifti. Görünen B dizisinin uzunluğu k için2^(−k); A ödülü0, B ödülleri sonuncu+1 olana kadar−1: bütün gözlenen istatistikler aynı. Normal denklemler sağda wA=−γ/(1+γ+γ²),wB=γ/[2(1+γ+γ²)] verir. Kitaptaki(−1/2,0) limit çifti B0 merkezlenmiş gösterimdir; ham indirimli çözümün limiti(−1/3,+1/6). γ<1 iken bu sabit kaydırma BE değerini korumaz. Kaynak iddiası ile düzeltme ayrı tutuldu; sağ BE minimumu γ↑1 için1/2 (hesaplandı).')
# Identifiability graph: node positions follow data/MDP/objective/minimizer levels.
def identity_graph(left):
 key='c11-id-left'if left else'c11-id-right';b=start(330,350,'Amaçların veriden belirlenebilirliği',key)+text(165,21,'MC amaçları'if left else'Bootstrap amaçları',14)
 if left:
  nodes={'D':(165,65,'Veri'),'M1':(65,125,'MDP₁'),'M2':(165,125,'MDP₂'),'VE1':(65,205,'VE₁'),'VE2':(165,205,'VE₂'),'RE':(275,205,'RE'),'opt':(165,300,'ortak w*')};edges=[('M1','D'),('M2','D'),('M1','VE1'),('M2','VE2'),('D','RE'),('VE1','opt'),('VE2','opt'),('RE','opt')]
 else:
  nodes={'D':(165,65,'Veri'),'M1':(40,125,'MDP₁'),'M2':(115,125,'MDP₂'),'BE1':(40,205,'BE₁'),'BE2':(115,205,'BE₂'),'PBE':(210,205,'PBE'),'TDE':(295,205,'TDE'),'o1':(40,300,'w₁*'),'o2':(115,300,'w₂*'),'o3':(210,300,'w₃*'),'o4':(295,300,'w₄*')};edges=[('M1','D'),('M2','D'),('M1','BE1'),('M2','BE2'),('D','PBE'),('D','TDE'),('BE1','o1'),('BE2','o2'),('PBE','o3'),('TDE','o4')]
 for a,z in edges:
  x,y,_=nodes[a];u,v,_=nodes[z];dx,dy=u-x,v-y;l=math.hypot(dx,dy);b+=line(x+dx/l*17,y+dy/l*12,u-dx/l*22,v-dy/l*14,arrow=key)
 for x,y,label in nodes.values():b+=text(x,y+4,label,12)
 return b+'</svg>'
fig(31,[identity_graph(True),identity_graph(False)],'Kaynak Şekil11.4 ilişkileri kendi hesaplanmış karşı örneklerimizin amaç adlarıyla gösterildi. Farklı MDP’ler aynı veri yasasını üretebilir. VE değerleri farklı olsa da RE aynı optimal parametre kümesini belirler. BE değerleri ve minimumları farklı olabilir; veri tek başına hangisini seçmez. PBE/TDE veriden belirlenebilir ve farklı minimumlar verebilir. Oklar tanımlama/üretme ilişkisini gösterir; veri gizli MDP’yi nedensel olarak yaratır iddiası değildir (modellerden türetildi).')
expected=E['baird']['profiles']['TDC']['history'];sampleRMS=[z[4]for z in case['sample_history']]
ss=weight_series(hist[1]);ss+=[dict(name='√VE',x=[z[0]for z in sample],y=sampleRMS,color=C[3]),dict(name='√PBE',x=[z['t']for z in expected],y=[z['RMSPBE']for z in expected],color=C[4],dash=True)]
# For the sampled PBE use actual sample weights and the full-row-rank projectionΠ=I.
ss[-1]['x']=[z['t']for z in hist[1]];ss[-1]['y']=[float(np.sqrt(np.mean((.99*(X@np.array(z['w']))[6]-X@np.array(z['w']))**2)))for z in hist[1]]
es=weight_series(expected)+[dict(name='√VE',x=[z['t']for z in expected],y=[z['RMSVE']for z in expected],color=C[3]),dict(name='√PBE',x=[z['t']for z in expected],y=[z['RMSPBE']for z in expected],color=C[4],dash=True)]
fig(32,[plot('TDC: örnek geçiş, tohum1',ss,'gerçek geçiş','ağırlık / RMS',(-3,11)),plot('TDC: beklenen güncelleme',es,'süpürme','ağırlık / RMS',(-3,11))],'Kaynak Şekil11.5:α0.005,β0.05,1000 geçiş/süpürme. Yardımcı başlangıcı0, kitapta verilen ρ(δ−uᵀx)x biçimi. Kendi beklenen sonuç √VE1.93546454 ve √PBE0.00737856: küçük artık, doğru değere henüz yakın olmak demek değildir. Kaynak metni VE yaklaşık2 derken grafik √VE gösterir; bu ayrım düzeltildi. Ağırlıklar0 olmak zorunda değil; tam çözüm yönü(1,1,1,1,1,1,4,−2). Öğrenme yavaş ama gürültüsüz hedefe yöneliktir (simüle edildi/hesaplandı).')
etd=E['baird']['profiles']['ETD']['history'];series=weight_series(etd)+[dict(name='√VE',x=[z['t']for z in etd],y=[z['RMSVE']for z in etd],color=C[3])];dist=sorted(S['ETD_distribution']['RMSVE']);n=len(dist)
fig(33,[plot('ETD: sonlu başlangıçla beklenen',series,'beklenen süpürme','ağırlık / RMS',(-8,12)),plot('Gerçek ETD:4096 koşunun dağılımı',[dict(name='en az bu hatadaki oran',x=[math.log10(z)for z in dist],y=[(n-i)/n for i in range(n)])],'log10(1000 adım sonu √VE)','üst kuyruk oranı',(0,1))],'Kaynak Şekil11.6 beklenen ETD karşılığı:α0.03,γ0.99,I1,m−1=0 ve m_t=d_b+γPπᵀm_(t−1). Ağırlıklar ve √VE için1000 beklenen güncelleme; son √VE4.58×10⁻¹³. Sağ panel aynı α/γ ile tohum1..4096 önceden seçilmiş1000’er gerçek örnek adımının tamamını gösterir: medyan √VE3840.0576, en büyük1.651654×10¹²;4096000 gerçek geçiş, kesilme0. Beklenen güncelleme eğrisi ile örneklenen ağırlıkların tam beklentisi özdeş sayılmaz. Kendi w8 en düşük−3.681685; kaynak çizimindeki daha büyük salınıma zorla uydurulmadı. Açık sonlu-m yinelememiz ve eşlikçi Python uygulaması aynı beklenen güncelleme düzenini kullanır. Yüksek takip-izi varyansı kaynakta da pratik güçlük olarak belirtilir; bu sabit-adım deneyi azalan-adım yakınsama teoremi değildir (hesaplandı/simüle edildi).')
(OUT/'ch11-etd.js').write_text('// Own source-parameter ETD runs: 4096 fixed seeds, 1000 real steps each.\nexport default '+json.dumps(S['ETD_distribution'],separators=(',',':'))+';\n')
print('Generated12 source-figure groups,22..33; minimum sampledETD RMS',dist[0])
