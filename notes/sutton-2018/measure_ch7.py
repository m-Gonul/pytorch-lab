"""Independent numerical reproduction of chapter7; output JSON path is argv1.
Portable Mulberry32 gives the same samples as the separate JS implementation.
"""
import math,json,time,sys
from pathlib import Path
class RNG:
 def __init__(self,seed=1):self.a=seed or 1
 def __call__(self):
  self.a=(self.a+0x6d2b79f5)&0xffffffff;t=((self.a^(self.a>>15))*(1|self.a))&0xffffffff;t=((t+(((t^(t>>7))*(61|t))&0xffffffff))^t)&0xffffffff;return ((t^(t>>14))&0xffffffff)/4294967296

def walks(N,R,E,seed=1):
 rng=RNG(seed);data=[]
 for k in range(R):
  run=[]
  for e in range(E):
   s=(N+1)//2;st=[s]
   while 0<s<N+1:s+=1 if rng()<.5 else -1;st.append(s)
   run.append(st)
  data.append(run)
 return data

def rms(V,N,left):return math.sqrt(sum((V[i]-(left+(1-left)*i/(N+1)))**2 for i in range(1,N+1))/N)

def learn_walk(run,n,alpha,N=19,left=-1,init=0,delta=False):
 V=[0]+[left+(1-left)*i/(N+1) if init=='true' else init for i in range(1,N+1)]+[0];errors=[]
 for st in run:
  T=len(st)-1;last=1 if st[-1] else left
  if not delta:
   for tau in range(T):
    h=min(tau+n,T);target=last if h==T else V[st[h]];s=st[tau];V[s]+=alpha*(target-V[s])
  else:
   prefix=[0.]
   # Only real transitions get a stored TD error; flush has none.
   for t in range(T+n-1):
    if t<T:prefix.append(prefix[-1]+(last if t==T-1 else V[st[t+1]])-V[st[t]])
    tau=t-n+1
    if tau>=0:V[st[tau]]+=alpha*(prefix[min(tau+n,T)]-prefix[tau])
  errors.append(rms(V,N,left))
 return errors,V

def grid(data,N=19,left=-1,init=0,ns=(1,2,4,8,16,32,64,128,256,512),delta=False):
 rows=[]
 for n in ns:
  for i in range(21):
   a=i/20;values=[]
   for run in data:
    hist,V=learn_walk(run,n,a,N,left,init,delta);values.append(sum(hist)/len(hist))
   mean=sum(values)/len(values);se=math.sqrt(sum((x-mean)**2 for x in values)/(len(values)-1)/len(values)) if len(values)>1 else 0
   rows.append(dict(n=n,alpha=a,mean=mean,se=se))
 return rows

def chain(R=1000,E=100,n=4,alpha=.1,p=.9,seed=1):
 rng=RNG(seed);sums=[[0.]*E for _ in range(2)];sq=[[0.]*E for _ in range(2)];runs=[]
 for r in range(R):
  tables=[[0.]*4 for _ in range(2)];last=[]
  for e in range(E):
   rho=[(p if rng()<.5 else 1-p)/.5 for _ in range(4)]
   for method,V in enumerate(tables):
    for tau in range(4):
     h=min(tau+n,4)
     if method==0:
      target=1 if h==4 else V[h];weight=math.prod(rho[tau:h]);V[tau]+=alpha*weight*(target-V[tau])
     else:
      target=0 if h==4 else V[h]
      for k in range(h-1,tau-1,-1):target=rho[k]*((1 if k==3 else 0)+target)+(1-rho[k])*V[k]
      V[tau]+=alpha*(target-V[tau])
    error=sum((x-1)**2 for x in V)/4;sums[method][e]+=error/R;sq[method][e]+=error*error/R
   last=[list(v) for v in tables]
  runs.append(last)
 return dict(mean_squared_error=sums,se_squared_error=[[math.sqrt(max(0,sq[j][e]-sums[j][e]**2)/(R-1)) for e in range(E)] for j in range(2)],last_run=runs[-1],last_allruns=runs)
def bellman_references():
 import numpy as np,runpy,contextlib,io
 root=Path(__file__).resolve().parents[2]
 with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path(str(root/'content/modules/m11-4-mc-td/x3.sol.py'))
 records=[]
 for N in [5,19]:
  for left in [-1.,0.]:
   for gamma in [0.,.5,.9,1.]:
    P=np.zeros((N,N));reward=np.zeros(N);reward[0]+=left/2;reward[-1]+=.5
    for k in range(N):
     if k:P[k,k-1]=.5
     if k<N-1:P[k,k+1]=.5
    true=np.linalg.solve(np.eye(N)-gamma*P,reward);actual=np.array(ns['walk_values'](N,left,gamma));error=float(np.max(abs(true-actual)));assert error<1e-12
    records.append(dict(N=N,left=left,gamma=gamma,V=true.tolist(),maximum_difference=error))
 return dict(records=records,x3_original_default_RMS=ns['nstep_td'](ns['episodes19'](10),4,.4))

def first_episode_example():
 out={}
 for n in [1,2,3,8]:
  V=[0]+[.5]*5+[0];path=[3,4,5,6];T=3
  for tau in range(T):
   h=min(tau+n,T);G=1 if h==T else V[path[h]];V[path[tau]]+=.1*(G-V[path[tau]])
  out[str(n)]=V
 return out

if __name__=='__main__':
 t=time.time();data=walks(19,100,10);source=grid(data);print('source best',min(source,key=lambda z:z['mean']),flush=True)
 comparison={str(n):grid(data,ns=[n],delta=True) for n in [1,4,8]};print('delta best',{k:min(v,key=lambda z:z['mean'])for k,v in comparison.items()},flush=True)
 other={};
 for N,left,init in [(19,0,.5),(19,0,0),(5,-1,0),(5,0,.5)]:
  key=f'{N}/{left}/{init}';other[key]=grid(data if N==19 else walks(N,100,10),N,left,init);print('variant best',key,min(other[key],key=lambda z:z['mean']),flush=True)
 ch=chain();print('chain first,lastMSE',[(z[0],z[-1])for z in ch['mean_squared_error']],flush=True)
 c1=chain(n=1);print('chain n1 maximum difference',max(abs(a-b)for a,b in zip(*c1['mean_squared_error'])),flush=True)
 out=dict(source_walk=dict(N=19,runs=100,episodes=10,seed=1,ns=[1,2,4,8,16,32,64,128,256,512],alphas=[i/20 for i in range(21)],rows=source,steps=sum(len(s)-1 for r in data for s in r)),delta_comparison=comparison,walk_variants=other,chain=dict(default=ch,n1=c1,parameters=dict(R=1000,E=100,n=4,alpha=.1,p=.9,b=.5,seed=1)))
 out['arithmetic']=dict(offpolicy_sarsa_correct=sum(.5*r*q for r,q in zip([1.5,.5],[2,4])),offpolicy_sarsa_missing_last=sum(.5*q for q in [2,4]),qsigma_correct_boundary=sum(.5*q for q in [2,4]),qsigma_printed_zero=[3-.5*q for q in [2,4]],single_IS_mean=sum(.5*r*reward for r,reward in zip([1.5,.5],[2,0])),double_IS_mean=sum(.5*r*r*reward for r,reward in zip([1.5,.5],[2,0])),action_dependent_baseline_mean=.5*(1-1.5),expected_exit_steps={str(N):((N+1)//2)**2 for N in [5,19]},chain_first_threshold=[next((i+1 for i,v in enumerate(a)if v<=.001),None)for a in ch['mean_squared_error']])
 out['bellman_references']=bellman_references();out['first_episode_example']=first_episode_example()
 Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print('seconds',time.time()-t)
