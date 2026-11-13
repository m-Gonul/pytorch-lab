from pathlib import Path
import math,json,sys

def txt(x,y,t,size=22):return f'<text x="{x}" y="{y}" text-anchor="middle" style="font-size:{size}px">{t}</text>'
def arrow(x,y,u,v):return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="currentColor" marker-end="url(#KEY)"/>'
def svg(w,h,body,key):return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{key}"><defs><marker id="{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor"/></marker></defs>{body.replace("KEY",key)}</svg>'
coords=[(75,65),(235,65),(395,65),(555,65),(555,190),(395,190),(235,190),(75,190)]
labels=['π₀','vπ₀','π₁','vπ₁','π₂','…','π*','v*'];b=''
for (x,y),label in zip(coords,labels):b+=txt(x,y+8,label,27)
for i,((x,y),(u,v)) in enumerate(zip(coords,coords[1:])):
 dx,dy=u-x,v-y;l=math.hypot(dx,dy);b+=arrow(x+dx/l*34,y+dy/l*34,u-dx/l*34,v-dy/l*34)+txt((x+u)/2+25*(dy!=0),(y+v)/2-12,['E','I','E','I','E/I','E/I','E'][i],22)
b+=txt(320,258,'E: politika değerlendirme; I: açgözlü iyileştirme',20)
f3='<figure class="fig wide">'+svg(640,285,b,'ch4-policy-sequence')+'<figcaption>Şekil 3 — E ve I adımlarıyla politika/değer zinciri. Mevcut iki durumlu el hesabında π₀=(kal,kal), Vπ₀=(0,4); π₁=(git,kal), Vπ₁=(3,4) hesaplanır ve sabitlenir. Ok grafiği bu güncelleme dizisinden üretildi. (Kaynak: s.80, numarasız politika yineleme zinciri.)</figcaption></figure>'
b=txt(35,130,'π',25)+txt(285,130,'V',25)+'<path d="M45,107 Q160,-2 275,107" fill="none" stroke="currentColor" marker-end="url(#KEY)"/><path d="M275,145 Q160,254 45,145" fill="none" stroke="currentColor" marker-end="url(#KEY)"/>'+txt(160,30,'değerlendirme',20)+txt(160,65,'V → vπ',20)+txt(160,215,'π → greedy(V)',20)+txt(160,249,'iyileştirme',20)+txt(160,285,'⋮',25)+txt(35,330,'π*',24)+txt(285,330,'v*',24)+arrow(60,317,260,317)+arrow(260,337,60,337)
left=svg(320,370,b,'ch4-gpi-loop')
A=(40.,50.);B=(270.,160.);C=(40.,270.);p=C;points=[p]
for i in range(8):
 start=A if i%2==0 else C;dx,dy=B[0]-start[0],B[1]-start[1];w=((p[0]-start[0])*dx+(p[1]-start[1])*dy)/(dx*dx+dy*dy);p=(start[0]+w*dx,start[1]+w*dy);points.append(p)
b=f'<path d="M40,50 L270,160 L40,270" fill="none" stroke="currentColor"/>'+txt(138,48,'V = vπ',20)+txt(138,290,'π = greedy(V)',20)+txt(270,144,'v*, π*',20)
for (x,y),(u,v) in zip(points,points[1:]):b+=arrow(round(x,2),round(y,2),round(u,2),round(v,2))
b+=txt(160,334,'eğik kısıtlara ardışık izdüşüm',17)+txt(160,357,'geometri benzetmesi, kanıt değil',17)
f4='<figure class="fig wide"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:12px">'+left+svg(320,370,b,'ch4-gpi-constraints')+'</div><figcaption>Şekil 4 — Değerlendirme ve iyileştirmenin ortak sabit noktası. Sol döngü her iki hedefi ve en iyi çifti gösterir. Sağ oklar iki eğik doğruya gerçekten hesaplanan izdüşümlerdir: tam adımlar PI benzetmesi, eksik adımlar GPI serbestliği. Gerçek politika/değer uzayı bu iki doğruyla özdeş değildir. (Kaynak: s.86–87, iki numarasız GPI şekli; yeniden türetildi.)</figcaption></figure>'

O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True)
(O/'f3.html').write_text(f3)
(O/'f4.html').write_text(f4)
(O/'projection_ch4.json').write_text(json.dumps(points,indent=2)+'\n')
print('f3.html, f4.html, projection_ch4.json')
