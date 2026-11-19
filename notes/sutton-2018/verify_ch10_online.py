"""Independent episode-wide-array n-step reference, deliberately no ring buffer.
Compare with compact Python/JS ring implementations through saved fixtures.
"""
from pathlib import Path
import math,json
BASE=Path(__file__).resolve().parent
class RNG:
 def __init__(self,seed):self.s=seed
 def draw(self):
  self.s=(self.s+0x6d2b79f5)&0xffffffff;t=self.s;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296

def feat(x,v,a):
 qx=math.floor(64*x/1.7);qv=math.floor(64*v/.14)
 return [((t*10+(qx+t)//8+6)*11+(qv+3*t)//8+4)*3+a for t in range(8)]
def ref(episodes,n,alpha,seed,budget):
 # Episode-wide states/actions. Length of completed episodes is never a budget cap.
 start,action=RNG(seed),RNG(seed+99173);w=[0.]*2640;lengths=[];physical=0
 def val(s,a):return sum(w[k]for k in feat(*s,a))
 def choose(s):
  qs=[val(s,a)for a in range(3)];ties=[a for a in range(3)if qs[a]==max(qs)];return ties[int(action.draw()*len(ties))]
 for ep in range(episodes):
  S=[(-.6+.2*start.draw(),0.)];A=[choose(S[0])];T=math.inf;t=0
  while True:
   cut=False
   if t<T:
    x,v=S[-1];v=max(-.07,min(.07,v+.001*(A[t]-1)-.0025*math.cos(3*x)));x=max(-1.2,min(.5,x+v));v=0. if x<=-1.2 else v;S.append((x,v));physical+=1
    if x>=.5:T=t+1
    else:A.append(choose(S[-1]));cut=physical>=budget
   tau=t-n+1
   if tau>=0:
    endpoint=min(tau+n,T);target=-float(endpoint-tau)
    if tau+n<T:target+=val(S[tau+n],A[tau+n])
    error=target-val(S[tau],A[tau]);inc=(alpha/8)*error
    for k in feat(*S[tau],A[tau]):w[k]+=inc
   if cut:return dict(w=w,lengths=lengths,actual_steps=physical,cutoff=True,unfinished_steps=t+1)
   if tau==T-1:lengths.append(int(T));break
   t+=1
  if physical>=budget and len(lengths)<episodes:return dict(w=w,lengths=lengths,actual_steps=physical,cutoff=True,unfinished_steps=0)
 return dict(w=w,lengths=lengths,actual_steps=physical,cutoff=False,unfinished_steps=0)

cases=[]
for n in[1,2,8,16]:
 q=dict(episodes=5,n=n,alpha=.5 if n==1 else .3,seed=1010001,budget=2000000);cases.append(dict(params=q,result=ref(**q)))
for budget in[1,8,1000]:
 q=dict(episodes=5,n=8,alpha=.3,seed=1010001,budget=budget);cases.append(dict(params=q,result=ref(**q)))
f=BASE/'measurements_ch10_independent.json';d=json.loads(f.read_text());d['online_episode_array_fixtures']=cases;f.write_text(json.dumps(d,indent=2)+'\n')
print([(c['params']['n'],c['params']['budget'],c['result']['lengths'])for c in cases])
