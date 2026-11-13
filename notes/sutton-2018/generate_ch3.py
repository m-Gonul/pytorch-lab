"""Bölüm 3 özgün şemaları ve bağımsız sayısal kontroller (NumPy gerekir)."""
from pathlib import Path
import html,json,math,sys
import numpy as np
O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True)
def esc(x):return html.escape(str(x))
def text(x,y,s,anchor='middle',size=16):return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px">{esc(s)}</text>'
def line(x,y,u,v,arrow=False):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="currentColor"'+(' marker-end="url(#ARROW)"' if arrow else '')+'/>'
def circle(x,y,s='',action=False):return f'<circle cx="{x}" cy="{y}" r="{5 if action else 12}" fill="{ "currentColor" if action else "var(--paper)"}" stroke="currentColor"/>'+ (text(x,y+5,s,size=14) if s else '')
def svg(w,h,aria,body,key):
    body=body.replace('ARROW','ch3-'+key)
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}"><defs><marker id="ch3-{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor"/></marker></defs>{body}</svg>'
def fig(n,panels,cap):
    body=panels[0] if len(panels)==1 else '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:12px">'+''.join(panels)+'</div>'
    (O/f'f{n}.html').write_text(f'<figure class="fig wide">{body}<figcaption>Şekil {n} — {cap}</figcaption></figure>')
# Interface: directed edges built from a timed interaction graph.
b=''
for x,y,label in [(240,45,'Ajan'),(240,225,'Ortam')]:b+=f'<rect x="{x}" y="{y}" width="160" height="65" rx="8" fill="var(--paper)" stroke="currentColor"/>'+text(x+80,y+39,label,size=22)
b+='<path d="M400,78 H575 V257 H402" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/><path d="M240,246 H85 V85 H237" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/><path d="M240,270 H130 V100 H237" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/>'
b+=line(240,246,190,246,True)+line(240,270,190,270,True)+'<path d="M178,233 V292" fill="none" stroke="currentColor" stroke-dasharray="4 4"/>'
b+=text(520,155,'eylem Aₜ',size=20)+text(48,165,'Sₜ',size=20)+text(167,175,'Rₜ',size=20)+text(82,305,'yeni durum Sₜ₊₁',anchor='start',size=18)+text(82,332,'yeni ödül Rₜ₊₁',anchor='start',size=18)
b+=text(320,375,'S₀ → A₀ → (R₁, S₁) → A₁ → (R₂, S₂) → …',size=19)
fig(3,[svg(640,400,'Ajanın eylemi, ortamın birlikte ürettiği yeni durum ve ödül',b,'interface')],'Ajan–ortam çevrimi, zaman indeksleriyle yeniden kuruldu. Eylem Aₜ sonrasında birlikte gelen büyüklükler Rₜ₊₁ ve Sₜ₊₁. (Kaynak: Sutton ve Barto, s.48, Şekil 3.1; özgün yönlü grafik.)')
# Recycling: two readable trees, all five actions and seven nonzero branches.
robot=[]
for state,rows in [('h',[('ara',[("h","α, rₛ"),("l","1−α, rₛ")]),('bekle',[("h","1, r𝓌")])]),('l',[('ara',[("h","1−β, −3"),("l","β, rₛ")]),('bekle',[("l","1, r𝓌")]),('şarj',[("h","1, 0")])])]:
    b=text(160,25,'Yüksek pil h' if state=='h' else 'Düşük pil l',size=20)
    ys=[92,245] if state=='h' else [75,200,315]
    for (name,outcomes),y in zip(rows,ys):
        b+=line(42,180,132,y,True)+text(89,(180+y)/2-9,name,size=16)
        for j,(dest,label) in enumerate(outcomes):
            yy=y+(j-(len(outcomes)-1)/2)*64
            b+=line(143,y,277,yy,True)+text(217,(y+yy)/2-9,label,size=15)+circle(290,yy,dest)
        b+=circle(138,y,action=True)
    b+=circle(30,180,state)
    b+=text(160,370,'○ durum; ● durum–eylem',size=17)
    robot.append(svg(320,395,'Geri dönüşüm robotunda '+state+' durumundan bütün geçişler',b,'robot-'+state))
fig(4,robot,'İki durum, beş eylem ve yedi pozitif olasılıklı dal. Aynı h/l etiketli yapraklar aynı durumu temsil eder; dallar, olasılık ve koşullu beklenen ödülle etiketlendi. Sıfır olasılıklı satırlar aşağıdaki tabloda ayrıca var. (Kaynak: s.52–53, numarasız robot diyagramı; geçiş listesinden üretildi.)')
# Cartpole geometry computed with sin/cos; absorbing-state sequence from rewards.
b=text(180,27,'Denge: konum x, açı θ',size=20)+line(15,250,345,250)
x,y=175,223;theta=.24;tip=(x+130*math.sin(theta),y-130*math.cos(theta))
b+=f'<rect x="{x-23}" y="{y}" width="46" height="20" fill="var(--paper)" stroke="currentColor"/>'+circle(x-14,y+22)+circle(x+14,y+22)+line(x,y,*tip)+line(x,y,x,y-130)
b+=text(212,170,'θ',size=19)+line(95,290,250,290,True)+text(180,317,'uygulanan kuvvet',size=18)+text(180,350,'Açı / ray sınırı → başarısızlık',size=17)
pole=svg(360,375,'Ray üzerindeki arabaya kuvvet uygulayarak çubuğu dik tutma',b,'pole')
b=text(180,27,'T = 3: sıfır ödüllü soğurucu',size=19)
for j in range(3):
    xx=30+j*97;b+=line(xx+13,115,xx+80,115,True)+text(xx+50,85,f'R{j+1} = +1',size=15)+circle(xx,115,f'S{j}')
b+='<rect x="309" y="103" width="24" height="24" fill="currentColor"/>'
b+='<path d="M326,104 C375,15 375,206 326,126" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/>'+text(180,200,'R₄ = R₅ = … = 0',size=20)+text(180,240,'+1, +1, +1, 0, 0, …',size=21)+text(180,285,'G₀ = 1 + γ + γ²',size=21)+text(180,330,'Sonlanınca kalan getiri 0.',size=18)
absorb=svg(390,375,'Üç ödülden sonra sıfır ödüllü soğurucu terminal duruma girme',b,'absorbing')
fig(5,[pole,absorb],'Çubuk dengesi geometrisi ve aynı sonlu bölümün sonsuz sıfır kuyruğuyla gösterimi. Sol çizim geometriden, sağ çizim ödül dizisinden üretildi. (Kaynak: s.56–57, iki numarasız şekil; kuvvet oku ve açı etiketi metni açıklamak için eklenmiştir.)')
# Synthetic finite golf graph: precise putter range 2, stochastic driver with two landing positions.
# Driver has two equally likely landing positions; only d=1 guarantees a hole-in-one.
N=15;T=0;P=np.zeros((N,2,N));rew=-np.ones((N,2));rew[0]=0
P[0,:,0]=1
for d in range(1,13):
    P[d,0,max(0,d-2)]=1
    targets=[0,0] if d==1 else [max(1,d-6),max(2,d-5)]
    for t in targets:P[d,1,t]+=.5
for s in [13,14]:
    P[s,0,s]=1;P[s,1,1]=.5;P[s,1,2]=.5
V=np.zeros(N)
for _ in range(100):
    Q=rew+P@V;W=Q.max(axis=1)
    if np.max(np.abs(W-V))<1e-13:V=W;break
    V=W
Q=rew+P@V
vp=[0]+[-math.ceil(d/2) for d in range(1,13)]
assert vp[12]==-6 and V[12]==-3 and Q[12,1]==-3
assert Q[13,0]==-3 and Q[13,1]==-2
panels=[]
for kind,title in [(0,'Yalnız putter: v_putt'),(1,'İlk driver, sonra en iyi: q*')]:
    b=text(180,25,title,size=18)
    # Distance bands, mapped geometrically from their exact graph values.
    for k in reversed(range(1,7 if kind==0 else 4)):
        rad=(k*25 if kind==0 else [0,17,88,155][k]);cy=205
        if kind==1 and k==1:b+=f'<ellipse cx="190" cy="205" rx="25" ry="26.5" fill="var(--c-p)" fill-opacity="0.23" stroke="currentColor"/>'
        b+=f'<ellipse cx="190" cy="{cy}" rx="{rad}" ry="{rad*1.06}" fill="{ "var(--c-p)" if k==1 and kind==0 else "none" if k==1 else "var(--paper)"}" fill-opacity="{.23 if k==1 else 1}" stroke="currentColor"/>'
        b+=text(190-rad*.70,205-rad*.72,f'−{k}',size=17)
    b+=text(190,220,'0',size=18)+line(190,191,190,165)+f'<path d="M190,165 L211,172 L190,179 Z" fill="currentColor"/>'
    b+=text(225,255,'green',size=16)+circle(30,205)+text(30,240,'tee',size=16)
    for sx,sy in [(235,100),(130,265)]:
        b+=f'<ellipse cx="{sx}" cy="{sy}" rx="30" ry="13" class="soft-b" stroke="currentColor"/>'+text(sx,sy-19,'kum',size=15)+text(sx,sy+5,'−∞' if kind==0 else '−2',size=15)
    if kind==1:
        b+=f'<ellipse cx="190" cy="205" rx="116" ry="122.96" fill="none" stroke="currentColor" stroke-dasharray="4 4"/>'+text(90,246,'−2.5',size=15)
    b+=text(180,392,'tee: 6 putt' if kind==0 else 'tee: 2 drive + 1 putt',size=18)
    b+=text(180,420,'putter kesin, menzil 2' if kind==0 else 'driver iki olası iniş',size=16)
    panels.append(svg(360,440,title+' için şematik mesafe bölgeleri ve kum engelleri',b,'golf-'+str(kind)))
fig(6,panels,'Golfte −1 vuruş maliyeti: putter sınırlı menzille kesin, driver uzun menzille değişken iniş sağlar. Delik 0; green −1; putter ile dış bölgeler −2…−6 ve kum −∞. Driver için −1 bölgesi daha küçük, −2 bölgesi green’e sonraki putt ile erişen yerleri, −3 bölgesi tee’yi içerir. Şekilde bölgeler şematiktir; doğrulanan 15 durumlu özgün modelde tee 6 putt veya 2 drive + 1 putt gerektirir. Kaynak sürekli saha geometrisini ve tüm iniş olasılıklarını vermediğinden bu çizim ona tekil sayısal bir model atfetmez. (Kaynak: s.61, Şekil 3.3; modelden hesaplanan değerlerle özgün çizim.)')
# Backup diagrams from branching descriptions: stochastic branches versus maximization.
backs=[]
for idx,(name,rootaction,short,optimal) in enumerate([
    ('vπ: eylem, sonra ortam',False,False,False),('qπ: ortam, sonra eylem',True,False,False),
    ('v*: en iyi eylem',False,False,True),('q*: ortam, en iyi devam',True,False,True),
    ('vπ = Σ π qπ',False,True,False),('qπ = Σ p (r + γvπ)',True,True,False)]):
    b=text(160,25,name,size=18);rx,ry=160,68
    xs=[60,160,260] if not rootaction or short else [88,232]
    for i,x in enumerate(xs):
        b+=line(rx,ry+7,x,160-10,True)
        b+=text((rx+x)/2+(12 if i==1 else 0),111,(('r'+['₁','₂','₃'][i]) if rootaction and short else 'p, r' if rootaction else 'max' if optimal and i==1 else 'π' if not optimal and i==1 else ''),size=16)
        b+=circle(x,160,action=not rootaction)
        b+=text(x,183,('s′'+str(i+1)) if rootaction else ('a'+str(i+1)),size=15)
        if short:b+=text(x,221,('vπ(s′'+str(i+1)+')') if rootaction else ('qπ(s,a'+str(i+1)+')'),size=15)
        if not short:
            for j,xx in enumerate([x-24,x,x+24] if idx==1 else [x-22,x+22]):
                b+=line(x,166,xx,235,True)+circle(xx,245,action=rootaction)
            b+=text(x,215,('max' if optimal else 'π') if rootaction else 'p, r',size=15)
            if i==len(xs)-1:b+=text(x+36,266,'a′' if rootaction else 's′',size=16)
    b+=circle(rx,ry,action=rootaction)+text(rx,51,'s, a' if rootaction else 's',size=16)
    if short:b+=text(260,72,'qπ(s,a)' if rootaction else 'vπ(s)',size=16)
    if optimal:
        # Arc convention at the agent choice, as in the source.
        if not rootaction:b+='<path d="M144,99 Q160,113 176,99" fill="none" stroke="currentColor" stroke-width="2"/>'
        else:
            for x in xs:b+=f'<path d="M{x-12},186 Q{x},197 {x+12},186" fill="none" stroke="currentColor" stroke-width="2"/>'
    b+=text(160,290,'○ durum; ● durum–eylem',size=16)
    backs.append(svg(320,315,name,b,'backup-'+str(idx)))
fig(7,backs,'Bakış bir adım açıldığında v kökü durum, q kökü durum–eylemdir. π etiketinde politika ortalaması, p/r etiketinde ortam beklentisi alınır. En iyilikte yalnız ajan dalında yayla gösterilen max vardır; ortam dalı beklenti kalır. Son iki panel Alıştırma 3.18–3.19 tek katmanlarını gösterir; son panelde r₁,r₂,r₃ koşullu beklenen ödüllerdir, durum dalları geçiş olasılıklarıyla ağırlıklandırılır. Yapraklar farklı çizilse de aynı durumu gösterebilir. (Kaynak: s.59,61–62 numarasız şemalar ve s.64, Şekil 3.4; dallanma kurallarından üretildi.)')
# Three-state example and exact values on both branches.
b=text(180,25,'Üst durumda tek seçim',size=20)
b+=line(180,75,82,145,True)+line(180,75,278,145,True)+text(110,104,'sol',size=18)+text(252,104,'sağ',size=18)
b+=line(82,150,82,215,True)+text(48,190,'+1',size=18)+line(278,150,278,215,True)+text(310,190,'0',size=18)
b+='<path d="M82,246 C0,315 0,35 168,63" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/><path d="M278,246 C360,315 360,35 192,63" fill="none" stroke="currentColor" marker-end="url(#ARROW)"/>'+text(20,300,'0',size=18)+text(337,300,'+2',size=18)
b+=line(82,240,82,246)+line(278,240,278,246)
for x,y,act in [(180,65,False),(82,145,True),(278,145,True),(82,228,False),(278,228,False),(82,246,True),(278,246,True)]:b+=circle(x,y,action=act)
b+=text(180,345,'sol: 1, 0, 1, 0, …',size=18)+text(180,375,'sağ: 0, 2, 0, 2, …',size=18)
three=svg(360,400,'Üç durumlu MDP: sol çevrimde 1 sonra 0, sağ çevrimde 0 sonra 2',b,'three')
b=text(180,25,'Politika değerleri (üst durum)',size=18)
for i,g in enumerate([0,.5,.9]):
    vl=1/(1-g*g);vr=2*g/(1-g*g)
    b+=text(180,95+i*90,f'γ = {g:g}: sol {vl:.3f}; sağ {vr:.3f}',size=18)+text(180,125+i*90,'sol' if vl>vr+1e-9 else 'sağ' if vr>vl+1e-9 else 'eşit; her karışım en iyi',size=18)
b+=text(180,380,'eşik: γ = 1/2',size=20)
fig(8,[three,svg(360,400,'İndirim sıfırda sol, yarımda eşit, onda dokuzda sağ',b,'three-values')],'İki deterministik politika için geometrik seri tam hesaplandı. Her iki dönüş oku üst duruma gelir. (Kaynak: s.66, Alıştırma 3.22 numarasız diyagramı; hesaplandı.)')
# Canonical grid independent NumPy solve and value iteration.
Pgrid=np.zeros((25,4,25));Rgrid=np.zeros((25,4))
for s in range(25):
    for a,(dr,dc) in enumerate([(-1,0),(1,0),(0,1),(0,-1)]):
        if s in [1,3]:t,r={1:(21,10),3:(13,5)}[s]
        else:
            y,x=divmod(s,5);y+=dr;x+=dc
            t,r=(s,-1) if not(0<=y<5 and 0<=x<5) else (y*5+x,0)
        Pgrid[s,a,t]=1;Rgrid[s,a]=r
Vrand=np.linalg.solve(np.eye(25)-.9*Pgrid.mean(axis=1),Rgrid.mean(axis=1))
Vo=np.zeros(25)
for k in range(10000):
    W=(Rgrid+.9*(Pgrid@Vo)).max(axis=1)
    if np.max(abs(W-Vo))<1e-12:Vo=W;break
    Vo=W
assert abs(Vo[1]-10/(1-.9**5))<1e-9
assert abs(Vrand[12]-.6731132598378817)<1e-12
shift=np.linalg.solve(np.eye(25)-.9*Pgrid.mean(axis=1),(Rgrid+1).mean(axis=1));assert np.max(abs(shift-Vrand-10))<1e-12
rewards=[-1,2,6,3,2];returns=[0.]*6
for t in reversed(range(5)):returns[t]=rewards[t]+.5*returns[t+1]
assert returns==[2,6,8,4,2,0]
# Independently enumerate six recycling policies at the stated example parameters.
a,b,rs,rw,g=.8,.5,1.,.2,.9
Pr=[[[a,1-a],[1,0]],[[1-b,b],[0,1],[1,0]]];Rr=[[rs,rw],[b*rs-3*(1-b),rw,0]]
pol=[]
for h in range(2):
    for l in range(3):
        pol.append(dict(actions=[h,l],V=np.linalg.solve(np.eye(2)-g*np.array([Pr[0][h],Pr[1][l]]),[Rr[0][h],Rr[1][l]]).tolist()))
rv=np.max([p['V'] for p in pol],axis=0);assert np.max(abs(rv-[8.474576271186445,7.627118644067801]))<1e-11
out=dict(grid_random=Vrand.tolist(),grid_optimal=Vo.tolist(),grid_optimal_actions=[np.flatnonzero(row>=row.max()-1e-9).tolist() for row in Rgrid+.9*(Pgrid@Vo)],grid_residual=float(np.max(abs(Vo-(Rgrid+.9*(Pgrid@Vo)).max(axis=1)))),returns=returns,ex3_9=dict(G1=7/(1-.9),G0=2+.9*7/(1-.9)),ex3_14=.9/4*(2.3+.4-.4+.7),ex3_24=10/(1-.9**5),robot_policies=pol,robot_optimal=rv.tolist(),golf=dict(putter=vp+['-infinity','-infinity'],optimal=V.tolist(),Q=Q.tolist(),driver_probabilities=P[:,1].tolist()))
(O/'measurements_ch3.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['returns','ex3_9','ex3_14','ex3_24','robot_optimal','grid_residual']},ensure_ascii=False))
