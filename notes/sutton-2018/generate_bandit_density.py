from pathlib import Path
import math,random
rng=random.Random(1);mu=[rng.gauss(0,1) for _ in range(10)]
body=''
def point(x,y):return f'{x:.2f},{y:.2f}'
def sy(y):return 30+(4.7-y)*26
for tick in range(-3,4):
    y=sy(tick);body+=f'<text x="32" y="{y+6}" text-anchor="end" style="font-size:18px">{tick}</text>'
body+='<path d="M45,152.2 H623" class="dash" stroke="currentColor" fill="none"/>'
for a,m in enumerate(mu):
    x=68+56*a
    ys=[m-3.1+i*6.2/120 for i in range(121)]
    left=[point(x-17*math.exp(-.5*(y-m)**2),sy(y)) for y in ys]
    right=[point(x+17*math.exp(-.5*(y-m)**2),sy(y)) for y in reversed(ys)]
    body+=f'<path d="M'+ ' L'.join(left+right)+' Z" fill="currentColor" fill-opacity="0.23"/>'
    body+=f'<path d="M{x-14},{sy(m)} H{x+14}" stroke="currentColor"/><text x="{x+4}" y="{sy(m)-7}" style="font-size:18px">q*({a+1})</text>'
    body+=f'<path d="M{x},285 v5" stroke="currentColor"/><text x="{x}" y="305" text-anchor="middle" style="font-size:18px">{a+1}</text>'
body+='<path d="M45,285 H623" stroke="currentColor" fill="none"/>'
body+='<text x="15" y="17" style="font-size:18px">ödül dağılımı</text><text x="320" y="331" text-anchor="middle" style="font-size:18px">eylem</text>'
fig='<figure class="fig wide"><svg viewBox="0 0 640 340" role="img" aria-label="On kolun birim varyanslı normal ödül dağılımları ve gerçek ortalamaları">'+body+'</svg><figcaption>Şekil 3 — Sutton ve Barto, Şekil 2.1 deney modelinin özgün çekilişle yeniden kurulması. q* değerleri N(0,1), her kolun ödülü N(q*,1); yoğunluklar formülden hesaplandı. Tohum 1 ile kendi görevimiz üretildi; kaynak şeklin rastgele sayıları kopyalanmadı. (hesaplandı)</figcaption></figure>'

import sys
Path(sys.argv[1]).write_text(fig)
