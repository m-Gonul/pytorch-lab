"""Own exact models and independent checks; OMP/BLAS threads should be1.
The 11-state model conditions on free servers before the next priority arrives.
An independently assembled44-state model checks its gain and differential values.
"""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import binom
BASE=Path(__file__).resolve().parent
REWARDS=np.array([1.,2.,4.,8.])

def kernel(p,include):
 D=np.zeros((11,2,11))
 for f in range(11):
  for a in range(2):
   after=f-min(a,f);busy=10-(after if include else f)
   for k in range(busy+1):D[f,a,after+k]=math.comb(busy,k)*p**k*(1-p)**(busy-k)
 assert np.max(abs(D.sum(2)-1))<1e-12
 return D

def evaluate(accept,D):
 policy=np.stack([1-accept,accept],axis=-1)
 P=np.einsum('fpa,fak->fk',policy,D)/4;r=accept@REWARDS/4
 A=np.column_stack([np.eye(11)-P,np.ones(11)]);A=np.vstack([A,np.r_[np.eye(1,11,0)[0],0]])
 sol=np.linalg.solve(A,np.r_[r,0]);h,rho=sol[:-1],float(sol[-1])
 B=P.T-np.eye(11);B[-1]=1;mu=np.linalg.solve(B,np.r_[np.zeros(10),1.])
 return rho,h,P,r,mu

def optimize(p,eps,include):
 D=kernel(p,include);accept=np.full((11,4),eps/2);accept[0]=0
 for iteration in range(100):
  rho,h,P,r,mu=evaluate(accept,D);adv=REWARDS[None,:]+((D[:,1]-D[:,0])@h)[:,None]
  greedy=(adv>0).astype(int);greedy[0]=0;new=eps/2+(1-eps)*greedy;new[0]=0
  if np.array_equal(new,accept):break
  accept=new
 else:raise AssertionError('policy iteration failed')
 q=np.zeros((11,4,2))
 for f in range(11):
  for j in range(4):
   for a in range(2):q[f,j,a]=(a*REWARDS[j]if f>0 else 0)-rho+D[f,a]@h
 # Independent44-state construction, independent SciPy binomial PMF and loops.
 P44=np.zeros((44,44));r44=np.zeros(44)
 for f in range(11):
  for j in range(4):
   row=f*4+j;r44[row]=accept[f,j]*REWARDS[j]
   for a in range(2):
    weight=accept[f,j]if a else 1-accept[f,j];after=f-min(f,a);busy=10-(after if include else f)
    for k in range(busy+1):P44[row,(after+k)*4:(after+k+1)*4]+=weight*binom.pmf(k,busy,p)/4
 assert np.max(abs(P44.sum(1)-1))<1e-12
 A44=np.column_stack([np.eye(44)-P44,np.ones(44)]);A44=np.vstack([A44,np.r_[np.eye(1,44,0)[0],0]])
 x=np.linalg.solve(A44,np.r_[r44,0]);expected_h=((1-accept)*q[:,:,0]+accept*q[:,:,1]).ravel()
 diff=float(max(abs(x[:-1]-expected_h)));gd=abs(float(x[-1])-rho)
 assert gd<1e-10 and diff<1e-9
 thresholds=[int(np.flatnonzero(greedy[:,j])[0])if np.any(greedy[:,j])else None for j in range(4)]
 return dict(release_new=include,epsilon=eps,rho=rho,h_before_priority=h.tolist(),accept=accept.tolist(),greedy=greedy.tolist(),Q=q.ravel().tolist(),mu_free=mu.tolist(),thresholds=thresholds,iterations=iteration+1,full44_gain=float(x[-1]),full44_value_max_difference=diff,full44_gain_difference=gd,full44_Bellman_residual=float(max(abs(r44+P44@x[:-1]-x[-1]-x[:-1]))))

models=[optimize(.06,eps,include)for include in [True,False]for eps in [0.,.1]]
P=np.roll(np.eye(3),1,axis=1);r=np.array([0.,0.,1.]);mu=np.ones(3)/3;rho=float(mu@r);h=np.linalg.solve(np.eye(3)-P+np.ones((3,1))*mu,r-rho)
theory=dict(ring_gain=rho,ring_bias=h.tolist(),ring_delta=(r-rho+P@h-h).tolist(),period2_gain=.5,period2_Abel_bias=[.25,-.25],gain_first_step_n2=dict(initial_gain=10,reward_sum=2,raw_delta=-18,naive_trace_gain=-8,correct_mean_target_gain=1),discount_counterexample=[dict(gamma=g,V_start_policy_A=2/(1-g),V_start_policy_B=4*g/((1-g)*(1+.5*g)),stationary_J_A=2/(1-g),stationary_J_B=(8/3)/(1-g))for g in [.1,.5,.9,.99]])
out=dict(model_assumptions='Independent per-busy-server Bernoulli releases; timing explicitly varied. Model is oracle, not shown to the learner.',models=models,theory=theory)
(BASE/'measurements_ch10_exact.json').write_text(json.dumps(out,indent=2)+'\n')

class RNG:
 def __init__(self,seed):self.a=seed
 def next(self):
  self.a=(self.a+0x6d2b79f5)&0xffffffff;t=self.a;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296

cpp=BASE/'measurements_ch10_access.json'
if cpp.exists():
 data=json.loads(cpp.read_text());results=[]
 for case in data['cases']:
  rgen=RNG(case['seed']);Q=np.zeros((11,4,2));gain=0.
  def action(f,j):
   if f==0:return 0
   if rgen.next()<.1:return int(rgen.next()*2)
   if Q[f,j,0]==Q[f,j,1]:return int(rgen.next()*2)
   return int(Q[f,j,1]>Q[f,j,0])
  f=10;j=int(rgen.next()*4);a=action(f,j);maxdiff=0.
  for row in case['first100']:
   reward=a*(1<<j);after=f-a;busy=10-(after if case['release_new']else f)
   nf=after+sum(rgen.next()<.06 for _ in range(busy));nj=int(rgen.next()*4);na=action(nf,nj)
   delta=reward-gain+float(Q[nf,nj,na])-float(Q[f,j,a]);gain+=.01*delta;Q[f,j,a]+=.01*delta
   own=[f,j,a,reward,nf,nj,na,delta,gain,float(Q[f,j,a])]
   assert own[:7]==row[:7];maxdiff=max(maxdiff,max(abs(x-y)for x,y in zip(own[7:],row[7:])))
   f,j,a=nf,nj,na
  assert maxdiff<1e-10 and sum(case['visits'])==2000000
  assert abs(sum(case['Q'])-case['rho'])<1e-8
  learned=np.array(case['Q']).reshape(11,4,2);accept=np.where(learned[:,:,1]>learned[:,:,0],.95,.05);accept[learned[:,:,1]==learned[:,:,0]]=.5;accept[0]=0
  actual_gain,*_=evaluate(accept,kernel(.06,case['release_new']))
  results.append(dict(seed=case['seed'],release_new=case['release_new'],first100_max_difference=maxdiff,final_behavior_model_gain=actual_gain,real_reward_mean=case['reward_mean'],last_gain_estimate=case['rho']))
 p=BASE/'measurements_ch10_independent.json';combined=json.loads(p.read_text())if p.exists()else{};combined['access_first100_and_behavior_gain']=results;p.write_text(json.dumps(combined,indent=2)+'\n')
 print('Eight access fixtures and learned-policy gains verified.')
print('11-state / independent44-state Poisson models agree; theory counterexamples computed.')
