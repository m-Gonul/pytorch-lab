from pathlib import Path
import json,runpy,time,sys
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

R=Path(__file__).resolve().parents[2]
M=R/'content/modules/m11-7-bandits'
out={}
# Enumerate exercise 2.2 before each action.
Q=np.zeros(4);N=np.zeros(4);rows=[]
for a,r in [(1,-1),(2,1),(2,-2),(2,2),(3,0)]:
    greedy=np.flatnonzero(Q==Q.max())+1
    rows.append(dict(Q=Q.tolist(),greedy=greedy.tolist(),action=a,forced_exploration=bool(a not in greedy)))
    N[a-1]+=1;Q[a-1]+=(r-Q[a-1])/N[a-1]
assert [i+1 for i,x in enumerate(rows) if x['forced_exploration']]==[4,5]
out['exercise2.2']=rows
m=quad(lambda x:10*x*np.exp(-x*x/2)/np.sqrt(2*np.pi)*ndtr(x)**9,-10,10)[0]
out['exercise2.3']=dict(expected_best=m,reward_eps001=.99*m,reward_eps01=.9*m,difference=.09*m)
# Arbitrary-step weights vs repeated update, independently check initial mass and normalization.
steps=np.array([.2,.3,.4,.1]);rewards=np.array([2.,-1,3,5]);initial=7.
v=initial
for a,r in zip(steps,rewards):v+=(r-v)*a
weights=np.array([steps[i]*np.prod(1-steps[i+1:]) for i in range(4)])
assert np.allclose(v,np.prod(1-steps)*initial+weights@rewards)
assert np.isclose(weights.sum()+np.prod(1-steps),1)
# Exercise 2.7 unbiased recency weights.
alpha=.1;v=7.;o=0
for r in rewards:
    o+=alpha*(1-o);v+=alpha/o*(r-v)
weights=alpha*(1-alpha)**np.arange(3,-1,-1)/(1-(1-alpha)**4)
assert np.allclose(v,weights@rewards)
out['weights']=dict(arbitrary_step=float(weights.sum()),bias_corrected_value=float(v))
# Baseline correction vs finite differences of a separate exact expectation.
H=np.array([.1,-.2,.4]);mu=np.array([4.1,3.7,5.]);b=2.3;t=5
def softmax(x):z=np.exp(x-x.max());return z/z.sum()
p=softmax(H)
g=np.array([(softmax(H+np.eye(3)[i]*1e-6)@mu-softmax(H-np.eye(3)[i]*1e-6)@mu)/2e-6 for i in range(3)])
updates=np.zeros(3)
for i in range(3):
    current=(1-1/t)*b+mu[i]/t
    updates+=p[i]*(mu[i]-current)*(np.eye(3)[i]-p)
assert np.allclose(updates,(1-1/t)*g,atol=1e-8)
assert not np.allclose(updates,g)
out['baseline']=dict(gradient=g.tolist(),expected_current_baseline_update=updates.tolist(),factor=1-1/t)
ns=runpy.run_path(str(M/'x2.sol.py'))
out['exercise2.5']={}
for alpha in [None,.1]:
    start=time.monotonic();curves=ns['nonstationary'](2000,10000,alpha,1,return_reward=True)
    out['exercise2.5'][str(alpha)]=dict(reward_last1000=float(curves['reward'][-1000:].mean()),optimal_last1000=float(curves['optimal'][-1000:].mean()),seconds=time.monotonic()-start)
    print('2.5',alpha,out['exercise2.5'][str(alpha)],flush=True)
configs=[]
for name,powers in [('epsilon',range(-7,-1)),('gradient',range(-5,3)),('ucb',range(-4,3)),('optimistic',range(-2,3)),('constant',range(-7,-1))]:
    configs.extend((name,2.**p) for p in powers)
fn=runpy.run_path(str(M/'x4.sol.py'))['bandit_sweep']
start=time.monotonic();values=fn(8,200000,configs,1)
out['exercise2.11']=dict(runs=8,steps=200000,interval=[100001,200000],seed=1,configs=configs,values=values.tolist(),seconds=time.monotonic()-start)
Path(sys.argv[1]).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('2.11',out['exercise2.11'],flush=True)
