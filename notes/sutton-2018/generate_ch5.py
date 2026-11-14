"""Original SVG diagrams computed from explicit toy returns/one-state models."""
from pathlib import Path
import sys,json
O=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/sutton-ch5');O.mkdir(parents=True,exist_ok=True)
def start(label,h):return f'<svg viewBox="0 0 640 {h}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="{label[:2]}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs>'
def txt(x,y,t,size=16):return f'<text x="{x}" y="{y}" text-anchor="middle" style="font-size:{size}px;fill:var(--ink)">{t}</text>'
def line(x,y,u,v):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>'
def circ(x,y,fill='var(--paper)',r=7):return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="var(--ink)"/>'
def fig(svg,n,cap):return f'<figure class="fig wide">{svg}</svg><figcaption>Şekil {n} — {cap}</figcaption></figure>'
r=[1,-2,4];g=.9;G=[0.]*3
for i in reversed(range(3)):G[i]=r[i]+(g*G[i+1] if i<2 else 0)
s=start('MC backup: durum ve eylem kökleri, tam tek yol',390)
for x,q in [(160,False),(475,True)]:
 s+=txt(x,27,'Q: kök (s,a)' if q else 'V: kök s')+circ(x,60,'var(--ink)' if q else 'var(--paper)')
 for j in range(3):
  y=60+j*78
  if not q:s+=line(x,y+7,x,y+20)+circ(x,y+24,'var(--ink)',4);y+=24
  s+=line(x,y+5,x,60+(j+1)*78-8)+txt(x+58,100+j*78,f'R{j+1}={r[j]}')
  yn=60+(j+1)*78
  if j==2:s+=f'<rect x="{x-8}" y="{yn-8}" width="16" height="16" fill="var(--paper)" stroke="var(--ink)"/>'+txt(x,yn+30,'terminal')
  else:s+=circ(x,yn)+ (line(x,yn+7,x,yn+20)+circ(x,yn+24,'var(--ink)',4) if q else '')
 s+=txt(x,361,f'γ={g}; G₀={G[0]:.2f}')
(O/'p3.html').write_text(fig(s,3,'MC-V ve Alıştırma5.3 MC-Q backup. Açık durum/siyah eylem/kare terminal; tek örnek yolun üç ödülüyle G₀ hesaplandı. Q için kök eylem zaten koşulludur. Kaynak s.95 numarasız V şemasının ilişkileri yeniden türetildi.'))
s=start('Tek durumlu sonsuz varyans MDP',260)+circ(285,100,r=22)+txt(285,105,'s')
s+='<path d="M270,82 C150,4 150,165 270,113" fill="none" stroke="var(--ink)" stroke-width="1.5"/>'+txt(143,55,'sol: 0.9, R=0')
s+=line(301,116,420,180)+line(310,97,520,97)
s+='<rect x="414" y="175" width="12" height="12" fill="var(--paper)" stroke="var(--ink)"/><rect x="520" y="91" width="12" height="12" fill="var(--paper)" stroke="var(--ink)"/>'
s+=txt(425,153,'sol: 0.1, R=+1')+txt(466,74,'sağ: 1, R=0')+txt(320,225,'π(sol)=1; b(sol)=b(sağ)=0.5; γ=1; vπ=1')
(O/'p4.html').write_text(fig(s,4,'Şekil5.4 modelinin yeniden hesaplanan MDP şeması. İki terminale giden ödüller ve döngü açık; E[X]=1, ikinci moment sonsuz (hesaplandı).'))
q=[0,1];pol0=0;pol1=max(range(2),key=lambda a:q[a])
s=start('MC Q politika GPI çevrimi ve tam değerlendirme zinciri',330)
s+=txt(128,90,'π',26)+txt(510,90,'Q',26)
s+='<path d="M150,70 Q320,10 490,70" fill="none" stroke="var(--ink)" stroke-width="2" marker-end="url(#MC-arrow)"/>'
s+='<path d="M490,110 Q320,190 150,110" fill="none" stroke="var(--ink)" stroke-width="2" marker-end="url(#MC-arrow)"/>'
s+=txt(320,22,'değerlendirme: Q → qπ')+txt(320,192,'iyileştirme: π → greedy(Q)')
labels=[f'π₀: a{pol0}',f'qπ₀=({q[0]},{q[1]})',f'π₁: a{pol1}=π*','qπ₁=q*']
for i,t in enumerate(labels):
 x=80+160*i;s+=txt(x,244,t,17)
 if i<3:s+=line(x+52,240,x+100,240).replace('stroke-width="1.5"','stroke-width="1.5" marker-end="url(#MC-arrow)"')+txt(x+80,224,'E' if i%2==0 else 'I')
s+=txt(320,295,'Tek durum, iki terminal eylem, ödüller 0 ve 1; γ=0.9')
(O/'c3.html').write_text(fig(s,3,'Kaynak s.97 içindeki Q–π GPI çevrimi ve E/I zinciri. Alttaki değerler tek-adımlı özgün MDP için tam hesaplandı; ideal zincirin bir sonlu örneği. Kaynağın numarasız iki şeması yeniden türetildi.'))
(O/'values.json').write_text(json.dumps(dict(rewards=r,gamma=g,returns=G,GPI=dict(Q=q,policies=[pol0,pol1])),indent=2))
