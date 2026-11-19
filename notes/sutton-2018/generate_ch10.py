"""Own chapter10 SVGs; no source pixels/curves. python generate_ch10.py OUTPUT."""
from pathlib import Path
import sys,json,math,html,hashlib
BASE=Path(__file__).resolve().parent;OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
M=json.loads((BASE/'measurements_ch10_mountain.json').read_text());A=json.loads((BASE/'measurements_ch10_access.json').read_text());E=json.loads((BASE/'measurements_ch10_exact.json').read_text())
C=['var(--c-a)','var(--c-b)','var(--c-p)','var(--c-q)','var(--ink)']
def text(x,y,s,size=12,anchor='middle',color='var(--ink)'):
 return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color}">{html.escape(str(s))}</text>'
def line(x,y,u,v,c='var(--ink)',dash=False,w=1.5):return f'<path d="M{x:.3f},{y:.3f} L{u:.3f},{v:.3f}" fill="none" stroke="{c}" stroke-width="{w}"'+(' stroke-dasharray="4 3"'if dash else'')+'/>'
def start(w,h,aria):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(aria,quote=True)}" style="width:100%;max-width:{w}px;flex:1 1 {w}px;height:auto">'
def fig(name,n,parts,cap):
 (OUT/name).write_text('<figure class="fig wide"><div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center">'+''.join(parts)+'</div><figcaption>Şekil '+str(n)+' — '+cap+'</figcaption></figure>')
def plot(title,series,xlabel,ylabel,ylim=None,xlim=None,log=False,ticks=None):
 w=330;h=345;L=50;R=317;T=50;B=235
 xs=[x for s in series for x in s['x']];ys=[y for s in series for y in s['y']if y is not None]
 lo,hi=xlim or(min(xs),max(xs));yl,yh=ylim or(min(0,min(ys)),max(ys)*1.04)
 X=lambda x:L+(R-L)*(x-lo)/(hi-lo)
 Y=lambda y:B-(B-T)*(math.log(y/yl)/math.log(yh/yl)if log else(y-yl)/(yh-yl))
 b=start(w,h,title)+text(165,22,title,14)+text(L,42,ylabel,11,'start')+line(L,T,L,B)+line(L,B,R,B)
 for y in (ticks or[yl+(yh-yl)*i/4 for i in range(5)]):b+=line(L,Y(y),R,Y(y),'var(--ink-3)',True,.5)+text(L-5,Y(y)+4,f'{y:g}',10,'end')
 for x in[lo,(hi+lo)/2,hi]:b+=text(X(x),B+17,f'{x:g}',10,'end'if x==hi else'middle')
 clip='ch10-'+hashlib.sha256(title.encode()).hexdigest()[:10]
 b+=f'<defs><clipPath id="{clip}"><rect x="{L}" y="{T}" width="{R-L}" height="{B-T}"/></clipPath></defs>'
 for k,s in enumerate(series):
  path='';go=False
  for x,y in zip(s['x'],s['y']):
   if y is None:go=False;continue
   path+=(' L'if go else' M')+f'{X(x):.3f},{Y(y):.3f}';go=True
  b+=f'<path clip-path="url(#{clip})" d="{path}" fill="none" stroke="{s.get("color",C[k%5])}" stroke-width="1.5"'+(' stroke-dasharray="4 3"'if s.get('dash')else'')+'/>'
  b+=line(12,280+13*k,31,280+13*k,s.get('color',C[k%5]),s.get('dash',False))+text(38,284+13*k,s['name'],10,'start')
 b+=text((L+R)/2,267,xlabel,11)
 return b+'</svg>'
# Own terrain cos(3x), physical boundaries and two impulse directions.
b=start(330,300,'Dağ arabası: yamaç, iki itiş, sol duvar ve sağ hedef')+text(165,22,'Motor tek başına eğime yetmez',14)
X=lambda x:35+(x+1.2)/1.7*255;Y=lambda x:110-55*math.sin(3*x)
path=' '.join(('M'if i==0 else'L')+f'{X(x):.2f},{Y(x):.2f}'for i in range(101)for x in[-1.2+1.7*i/100]);b+=f'<path d="{path}" fill="none" stroke="var(--ink)" stroke-width="2"/>'
x=-.52;cx=X(x);cy=Y(x)-9;b+=f'<rect x="{cx-10}" y="{cy-7}" width="20" height="10" fill="var(--c-a)"/>'
for off in[-6,6]:b+=f'<circle cx="{cx+off}" cy="{cy+5}" r="3" fill="var(--ink)"/>'
b+=line(60,190,140,190,C[1])+text(100,210,'önce geri',12)+line(170,190,275,190,C[0])+text(223,210,'sonra ileri',12)
b+=line(X(.5),Y(.5),X(.5),Y(.5)-35)+text(278,80,'hedef +0.5',11)+text(60,112,'−1.2 duvar',11)+text(165,246,'A = −1 / 0 / +1; her geçiş −1',12)+text(165,272,'Başlangıç x∈[−0.6,−0.4), hız 0',11)
b+='<path d="M60,190 L68,186 L68,194 Z" fill="var(--c-b)"/><path d="M275,190 L267,186 L267,194 Z" fill="var(--c-a)"/>'
parts=[b+'</svg>']
for frame in M['surfaces']['frames']:
 z=frame['grid_cost'];peak=max(z);proj=lambda i,j:(165+105*(i/40-j/40),155+45*(i/40+j/40)-95*z[j*41+i]/max(1,peak))
 label=frame['label'].replace('Episode','Bölüm ').replace('Step','Adım ')
 b=start(330,300,label+' maliyet yüzeyi, konum ve hız')+text(165,20,label,14)
 for j in range(0,41,4):
  path=' '.join(('M'if i==0 else'L')+f'{proj(i,j)[0]:.2f},{proj(i,j)[1]:.2f}'for i in range(41));b+=f'<path d="{path}" fill="none" stroke="{C[0]}" stroke-width=".7"/>'
 for i in range(0,41,4):
  path=' '.join(('M'if j==0 else'L')+f'{proj(i,j)[0]:.2f},{proj(i,j)[1]:.2f}'for j in range(41));b+=f'<path d="{path}" fill="none" stroke="{C[1]}" stroke-width=".7"/>'
 b+=text(165,42,f'−max Q: tepe {peak:.5f}',11)+text(244,247,'konum →',11)+text(83,247,'← hız',11)+text(165,270,'x: −1.2..+0.5; v: −0.07..+0.07',10)+text(165,291,f'{frame["steps"]} gerçek geçiş',11)
 parts.append(b+'</svg>')
fig('mc18.html',18,parts,'Kaynak Şekil 10.1 karşılığı: kendi fizik şemamız ve beş gerçek maliyet yüzeyimiz. Step 428 henüz ilk bölümün içindedir; diğerleri tamamlanan bölüm sayısıdır. λ=0.9, replacing iz ve toplam adım 0.3 bizim seçimimizdir; kaynağın dipnotu SARSA(λ) der, bu ayarları vermez. Yazar koşusunun tepe ölçekleri 4/27/46/104/120; bizim tepelerimiz 13.51926/97.81987/119.71176/107.26110/122.14259. Yükseklikler uydurulmadı; her panel kendi açık tepe ölçeğiyle izdüşürülür. Sağ hedef düzlemi maliyet 0; kaynak çizimindeki +0.6 işareti fizik hedefi +0.5 değerini değiştirmez (simüle edildi).')
xx=list(range(1,501))
fig('mc19.html',19,[plot('Tek-adım: 100 koşu × 500 bölüm',[dict(name='α='+str(v['alpha_times8'])+'/8',x=xx,y=v['results']['mean'])for v in M['full'][:3]],'bölüm','hedefe kadar adım (log)',(100,4000),log=True,ticks=[100,200,400,1000,4000])],'Kaynak Şekil 10.2 karşılığı; ε=0, Q başlangıcı 0, her eşitlikte rastgele eylem. Sekiz döşeme ve aktif ağırlık başına α. İlk bölüm ortalamaları 3390.91/2609.41/1465.07, son bölüm 138.86/124.76/122.07. Her eğri 100×500 gerçek hedefe varışı içerir; tohumlar kaynakta verilmediğinden yazar eğrileriyle noktasal eşitlik iddiası yok (simüle edildi).')
fig('mc20.html',20,[plot('Tek-adım / 8-adım',[dict(name=f'n={v["n"]}, α={v["alpha_times8"]}/8',x=xx,y=v['results']['mean'])for v in M['full'][2:]],'bölüm','hedefe kadar adım (log)',(100,2000),log=True,ticks=[100,200,400,1000,2000])],'Kaynak Şekil 10.3 karşılığı: 100 koşu×500 bölüm, n=1 için toplam adım 0.5, n=8 için 0.3. Son bölüm ortalamaları 122.07 ve 110.13. Bu ölçüm orta ufkun yararını gösterir; n büyüdükçe her görevde iyileşme teoremi değildir (simüle edildi).')
ser=[]
for n in[1,2,4,8,16]:
 cs=[v for v in M['early']if v['n']==n];ser.append(dict(name='n='+str(n),x=[v['alpha_times8']for v in cs],y=[v['mean_first50']for v in cs]))
fig('mc21.html',21,[plot('100 koşu × ilk 50 bölüm',ser,'toplam adım 8α','ortalama bölüm adımı',(210,600)),plot('Kaynak görünümü: 220..300',ser,'toplam adım 8α','ortalama bölüm adımı',(220,300),ticks=[220,240,260,280,300])],'Kaynak Şekil 10.4 karşılığı. α noktaları geometrik seçildi; x ekseni doğrusal ve noktalar doğruyla bağlandı. Kaynak görünümü yaklaşık 220..300 aralığını vurgular; solda kötü ayarları gizlememek için y 210..600, sağda 220..300 yakın görünümü. Kendi en iyi tamamlanmış ayarımız n=4, 8α=0.7430084: 222.4388±0.8964231 SE. n=4/8α=1.540702 ayarı 99/100 koşu tamamladı; kalan koşunun son bölümünde 2m adım sınırı, ortalama/SE boş. Kaynağın n=1 için yaklaşık 0.5 ve n=16 için 4 SE sayıları yazar raporudur; kendi sayılarımız etkileşimli alanda seçilir (simüle edildi).')
# Average-reward own figures, numbered independently in the new module.
fig('avg1.html',1,[plot('Üç durum: beklenen merkezli toplam',[dict(name='başlangıç '+name,x=list(range(13)),y=[sum([0,0,1][(s+t)%3]-1/3 for t in range(T))for T in range(13)])for s,name in enumerate('ABC')],'ufuk','E[Σ(R−1/3)]',(-.7,.7)),plot('İki durum: Abel değerleri',[dict(name='A: 1/[2(1+γ)]',x=[i/1000 for i in range(1000)],y=[1/(2*(1+i/1000))for i in range(1000)]),dict(name='B: −1/[2(1+γ)]',x=[i/1000 for i in range(1000)],y=[-1/(2*(1+i/1000))for i in range(1000)])],'γ','sonsuz ağırlıklı beklenti',(-.55,.55))],'Kaynak Alıştırma 10.6/10.7: üçlü halkanın ham beklenen toplamı salınır; Poisson çözümü µᵀh=0 ile (−1/3,0,+1/3). İki durumdaki Abel limitleri +1/4 ve −1/4; önce ufuk sonsuz, sonra γ↑1. Örnek yolun merkezli sonsuz toplamının var olduğu varsayılmadı (formüllerden hesaplandı).')
d=next(v for v in A['cases']if v['seed']==1 and v['release_new']);Q=d['Q'];b=start(330,300,'Öğrenilen açgözlü kabul politikası; öncelik ve boş sunucu')+text(165,20,'2m geçiş, tohum 1',14)
for j in range(4):
 b+=text(30,70+38*(3-j),str(2**j),12)
 for f in range(1,11):
  k=(f*4+j)*2;equal=Q[k]==Q[k+1];accept=Q[k+1]>Q[k];fill='var(--ink-3)'if equal else C[0]if accept else'var(--paper)';b+=f'<rect x="{46+(f-1)*27}" y="{48+(3-j)*38}" width="25" height="30" fill="{fill}" stroke="var(--ink-3)"/>'
for f in range(1,11):b+=text(58+(f-1)*27,218,f,11)
b+=text(177,242,'boş sunucu; 0 ise zorunlu red',11)+text(165,266,'renk: kabul / boş: red / gri: eşit',11)+text(165,288,'öncelik yukarı doğru 1,2,4,8',11)
vals=[dict(name='öncelik '+str(2**j),x=list(range(11)),y=[Q[(f*4+j)*2]if not f else max(Q[(f*4+j)*2:(f*4+j)*2+2])for f in range(11)])for j in range(4)]
fig('avg2.html',2,[b+'</svg>',plot('İzinli eylemlerin en iyi ham Q değeri',vals,'boş sunucu','diferansiyel Q, kaydırılmadı',(-11,13)),plot('Ziyaret: en yüksek boşluklar seyrek',[dict(name='öncelik '+str(2**j),x=list(range(11)),y=d['visits'][j::4])for j in range(4)],'boş sunucu','durum ziyareti',(0,150000))],'Kaynak Şekil 10.5 karşılığı, kendi 2m adımlık diferansiyel SARSA koşumuz: α=β=0.01, ε=0.1; yeni kabul edilmiş iş de aynı adımda boşalabilir, boşalmalar bağımsız Bernoulli p=0.06. Son Rbar 2.60472981, gerçekleşen ortalama 2.6349205. Kaynak yaklaşık 2.31 ve öncelik 1/2/4/8 için kabul eşiği 6/4/2/1 raporlar; bunları kendi ölçümümüz diye çizmedik. f=10 durumunda ziyaretler 0/0/1/0: sağ uçtaki düşüş ve eşitlikler yetersiz veriye bağlıdır. f=0 için geçersiz kabul Q=0 değerini en iyi eylem saymadık. Diferansiyel değerler keyfî bir sabite kadar tanımlıdır; burada öğrenilmiş ham tablo kullanıldı (simüle edildi).')
xs=[i/100 for i in range(100)]
fig('avg3.html',3,[plot('Aynı başlangıç S0',[dict(name='A: 2',x=xs,y=[2]*100),dict(name='B: 4γ/(1+γ/2)',x=xs,y=[4*g/(1+.5*g)for g in xs])],'γ','(1−γ)Vγ(S0)',(0,3)),plot('Kendi durağan dağılımları',[dict(name='A: ρ=2',x=xs,y=[2]*100),dict(name='B: ρ=8/3',x=xs,y=[8/3]*100)],'γ','(1−γ)ΣµπVγ',(0,3))],'Kendi iki durumlu karşı örneğimiz. S0 kal ödülü 2; diğer eylem 0 ile S1; S1 ödülü 4 ve yarım olasılıkla S0. Sabit başlangıç amacı γ=2/3 noktasında politika sırasını değiştirir. Kendi durağan dağılımındaki amaç her γ için ortalama ödülle aynı sıralamayı verir. Kaynak s.254 kanıtının µπ koşulu görünürdür; genel indirimli kontrol olanaksızlığı çıkmaz (hesaplandı).')
raw=naive=proper=10.;o=0.;curves=[[10.],[10.],[10.]]
for t in range(20):
 o+=.1*(1-o);raw+=.1*(2-2*raw);naive+=.1/o*(2-2*naive);proper+=.1/o*(1-proper)
 for v,c in zip([raw,naive,proper],curves):c.append(v)
fig('avg4.html',4,[plot('n=2; her ortalama hedefi 1',[dict(name=name,x=list(range(21)),y=y)for name,y in zip(['ham βδ','yanlış (β/o)δ','doğru (β/o)(δ/n)'],curves)],'güncelleme','ortalama ödül kestirimi',(-10,11))],'Kaynak Alıştırma 10.9 için n ölçeğinin etkisi: Rbar başlangıcı 10, β=0.1, ilk δ=−18. Ham güncelleme 8.2; naif iz −8; doğru ortalama hedefiyle iz 1. Bu başlangıç katsayısı hesabıdır; öğrenilen Q/politika altında gerçek gain için yansızlık teoremi değildir (hesaplandı).')

# Compact browser data shares the measured curves; weights remain in the audit JSON.
small={k:v for k,v in M.items()if k not in ['surfaces','fixture_n1_alpha_point5_first50']}
small['surfaces']={k:v for k,v in M['surfaces'].items()if k!='frames'}
small['surfaces']['frames']=[{k:v for k,v in f.items()if k!='weights'}for f in M['surfaces']['frames']]
(OUT/'ch10-mountain.js').write_text('// Own full-budget simulations. Reproduce with notes/sutton-2018/measure_ch10_mountain.cpp.\nexport default '+json.dumps(small,separators=(',',':'))+';\n')
