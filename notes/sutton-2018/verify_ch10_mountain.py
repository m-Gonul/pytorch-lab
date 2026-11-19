import math,json,numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parent
D=json.loads((BASE/'measurements_ch10_mountain.json').read_text())
class RNG:
 def __init__(self,seed):self.a=seed
 def next(self):
  self.a=(self.a+0x6d2b79f5)&0xffffffff;t=self.a;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296

def base(x,v):
 # Author API quantizes num_tilings*scaled_float, then shifts odd offsets.
 u=math.floor(8*(8*x/1.7));z=math.floor(8*(8*v/.14))
 return [((t*10+(u+t)//8+6)*11+(z+3*t)//8+4)*3 for t in range(8)]
def Q(w,f,a):return sum(float(w[i+a])for i in f)
def act(w,f,r):
 qs=[Q(w,f,a)for a in range(3)];m=max(qs);ties=[a for a in range(3)if qs[a]==m];return ties[int(r.next()*len(ties))]
def step(x,v,a):
 vp=min(.07,max(-.07,v+.001*(a-1)-.0025*math.cos(3*x)));xp=min(.5,max(-1.2,x+vp));return xp,(0. if xp<=-1.2 else vp)
start=RNG(1010001);r=RNG(1010001+99173);w=[0.]*2640;lengths=[]
for ep in range(50):
 x=-.6+.2*start.next();v=0.;f=base(x,v);a=act(w,f,r)
 for t in range(2000000):
  xp,vp=step(x,v,a);fp=base(xp,vp);done=xp>=.5;ap=0 if done else act(w,fp,r);delta=-1+(0 if done else Q(w,fp,ap))-Q(w,f,a)
  for i in f:w[i+a]+=.5/8*delta
  if done:lengths.append(t+1);break
  x,v,f,a=xp,vp,fp,ap
assert lengths==D['fixture_n1_alpha_point5_first50'],(lengths,D['fixture_n1_alpha_point5_first50'])
print('Independent direct one-step matches all first50 episode lengths.')
maxcost=0.
for frame in D['surfaces']['frames']:
 ws=frame['weights'];cost=[]
 for j in range(41):
  for i in range(41):
   f=base(-1.2+1.7*i/40,-.07+.14*j/40);cost.append(0. if i==40 else -max(Q(ws,f,a)for a in range(3)))
 diff=max(abs(a-b)for a,b in zip(cost,frame['grid_cost']));maxcost=max(maxcost,diff)
 assert diff<1e-9,diff
print('All5 cost fields, independent feature quantization maxdifference',maxcost)
# Independent NumPy trace updates, same policy arithmetic, to first12episodes.
w=np.zeros(2640);start=RNG(1110001);r=RNG(1209174);steps=0;checks=[]
for ep in range(12):
 trace=np.zeros(2640);x=-.6+.2*start.next();v=0.;f=base(x,v);a=act(w,f,r)
 for t in range(2000000):
  xp,vp=step(x,v,a);fp=base(xp,vp);done=xp>=.5;ap=0 if done else act(w,fp,r);delta=-1+(0 if done else Q(w,fp,ap))-Q(w,f,a);trace[np.array(f)+a]=1;w+=(.3/8*delta)*trace;trace*=.9;steps+=1
  if steps==428:
   diff=float(max(abs(w-np.array(D['surfaces']['frames'][0]['weights']))));assert diff<1e-9;checks.append(diff)
  if done:break
  x,v,f,a=xp,vp,fp,ap
assert steps==D['surfaces']['frames'][1]['steps'];diff=float(max(abs(w-np.array(D['surfaces']['frames'][1]['weights']))));assert diff<1e-9;checks.append(diff)
dest=BASE/'measurements_ch10_independent.json';out=json.loads(dest.read_text())if dest.exists()else{};out.update(first50_lengths_identical=True,cost_field_max_difference=maxcost,trace_weight_max_differences=checks);dest.write_text(json.dumps(out,indent=2)+'\n');print('Independent first12 trace episodes, weightdiffs',checks)
