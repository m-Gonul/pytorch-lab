from functools import lru_cache
from pathlib import Path
import json
import sys

W=Path(sys.argv[1]); W.mkdir(parents=True,exist_ok=True)
lines=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
def winner(b):
    for ids in lines:
        if b[ids[0]]!='.' and len({b[i] for i in ids})==1:return b[ids[0]]
    return None
def put(b,i,turn):return b[:i]+turn+b[i+1:]
@lru_cache(None)
def value(b,turn):
    w=winner(b)
    if w or '.' not in b:return float(w=='X')
    vs=[value(put(b,i,turn),'O' if turn=='X' else 'X') for i,c in enumerate(b) if c=='.']
    return max(vs) if turn=='X' else sum(vs)/len(vs)
def moves(b,turn):
    return [(value(put(b,i,turn),'O' if turn=='X' else 'X'),i,put(b,i,turn)) for i,c in enumerate(b) if c=='.']

a='....X....'
trace=None
for _,ib,b in moves(a,'O'):
    c=max(moves(b,'X'))[2]
    if winner(c):continue
    for _,idd,d in moves(c,'O'):
        if winner(d):continue
        opts=moves(d,'X');best=max(x[0] for x in opts)
        for ve,ie,e in opts:
            if ve>=best or winner(e):continue
            for _,iff,f in moves(e,'O'):
                if winner(f):continue
                g=max(moves(f,'X'))[2]
                trace=[a,b,c,d,e,f,g]
                break
            if trace:break
        if trace:break
    if trace:break
assert trace
(W/'ch1-trace.json').write_text(json.dumps(dict(boards=trace,values=[value(b,'O' if i%2==0 else 'X') for i,b in enumerate(trace)]),indent=2))

def fig(n,h,aria,body,cap):
    return f'<figure class="fig wide"><svg viewBox="0 0 640 {h}" role="img" aria-label="{aria}">{body}</svg><figcaption>Şekil {n} — {cap}</figcaption></figure>'

board='XOOOXX..X'
assert winner(board)=='X'
body='<text x="20" y="25" class="sm" style="font-size:18px">X çaprazı tamamladı: sonuç = 1</text>'
for x in (291,351):body+=f'<path d="M{x},42 V222" class="ink w2" stroke="currentColor"/>'
for y in (102,162):body+=f'<path d="M231,{y} H411" class="ink w2" stroke="currentColor"/>'
for i,c in enumerate(board):
    if c!='.':body+=f'<text x="{261+60*(i%3)}" y="{82+60*(i//3)}" text-anchor="middle" fill="currentColor" style="font-size:25px">{c}</text>'
body+='<path d="M246,57 L396,207" class="ca w3" stroke="currentColor" fill="none"/>'
(W/'board.html').write_text(fig(2,242,'Üç çarpı ile kazanılan kaynak XOX konumu',body,'XOX kuralları, kaynak s. 8 içindeki numarasız tahta. Aynı hücre dizilimiyle kazanan çapraz programla doğrulandı (hesaplandı).'))

pts=[(300,35),(300,87),(258,139),(258,191),(387,243),(387,295),(451,347)]
body='<defs><marker id="sutton-ch1-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#d34c4c"/></marker></defs>'
body+='<text x="300" y="17" text-anchor="middle" class="sm" style="font-size:18px">başlangıç konumu</text>'
for i,((x,y),(xx,yy)) in enumerate(zip(pts,pts[1:])):
    body+=f'<path d="M{x},{y} L{xx},{yy}" stroke="currentColor" fill="none"/>'
    label='rakibin hamlesi' if i%2==0 else 'bizim hamlemiz'
    body+=f'<text x="19" y="{(y+yy)/2+4}" class="sm" style="font-size:18px">{label}</text><path d="M170,{y+8} Q160,{y+8} 165,{(y+yy)/2} Q160,{yy-8} 170,{yy-8}" class="ink w2" stroke="currentColor" fill="none"/>'
for i in (1,3,5):
    x,y=pts[i]
    for dest in (x-100,x-52,x+54,x+105):
        if abs(dest-pts[i+1][0])>10:body+=f'<path d="M{x},{y} L{dest},{y+46}" class="dash" stroke="currentColor" fill="none"/>'
for i,(x,y) in enumerate(pts):
    label='abcdefg'[i]+(' = c*' if i==2 else ' = g*' if i==6 else '')
    body+=f'<circle cx="{x}" cy="{y}" r="3.8" fill="currentColor"/><text x="{x+10}" y="{y+5}" class="sm" style="font-size:18px">{label}</text>'
body+='<circle cx="206" cy="237" r="4" stroke="currentColor" fill="none"/><text x="184" y="257" class="sm" style="font-size:18px">e*</text>'
body+='<path d="M258,135 C210,95 242,53 295,37" fill="none" stroke="#d34c4c" stroke-width="1.6" marker-end="url(#sutton-ch1-arrow)"/><path d="M451,343 C525,286 468,247 393,243" fill="none" stroke="#d34c4c" stroke-width="1.6" marker-end="url(#sutton-ch1-arrow)"/>'
body+='<text x="420" y="379" class="sm" style="font-size:18px">… devam edebilir</text><text x="19" y="412" class="sm" style="font-size:18px">düz: oynanan · kesikli: denenmeyen · kırmızı: değer aktarımı</text>'
(W/'tree.html').write_text(fig(3,430,'a ile g arasındaki XOX hamleleri, keşif ve iki geriye öğrenme oku',body,'Sutton ve Barto, Şekil 1.1 ilişkilerinin yeniden kurulması: c → a, g → e; e keşif hamlesi, e* daha yüksek değerli alternatif. Dallar yasal oyun ağacı üzerinde denetlendi; bir örnek yörünge gerçek kazanma olasılıklarıyla seçildi (hesaplandı). Geometri kaynak görselden alınmadı.'))
print('trace',trace)
print('values',[value(b,'O' if i%2==0 else 'X') for i,b in enumerate(trace)])
