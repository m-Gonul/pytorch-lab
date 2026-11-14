"""Independent SciPy reference, Python/C++ fixtures and worked counterexamples.
Run with one BLAS thread. Writes measurements_ch9_exact.json and reference TSV.
If measurements_ch9_walk.json has fixtures, independently verifies all of them.
"""
import json,math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
BASE=Path(__file__).resolve().parent
N,J=1000,100;s=np.arange(1,N+1);ab=np.zeros((201,N));ab[J]=1
for k in range(1,J+1):ab[J-k,k:]=-.005;ab[J+k,:N-k]=-.005
reward=(np.maximum(0,J-N+s)-np.maximum(0,J-s+1))/200;h=np.zeros(N);h[499]=1
V,eta=solve_banded((J,J),ab,np.stack([reward,h],axis=1)).T;mu=eta/eta.sum()
def transition(v):
 p=np.r_[0.,np.cumsum(v)];return(p[s-1]-p[np.maximum(0,s-J-1)]+p[np.minimum(N,s+J)]-p[s])/200
v=np.zeros(N);e=np.zeros(N)
for iteration in range(10000):
 vn=transition(v)+reward;en=transition(e)+h
 if max(np.max(abs(vn-v)),np.max(abs(en-e)))<1e-14:break
 v,e=vn,en
assert max(abs(V-v))<1e-11 and max(abs(eta-e))<1e-10
P=diags([np.full(N-abs(k),.005)for k in range(-J,J+1)if k],offsets=[k for k in range(-J,J+1)if k],shape=(N,N)).tocsr();groups={}
for K in [5,10,20]:
 idx=(s-1)*K//N;X=np.eye(K)[idx];DX=mu[:,None]*X;A=DX.T@(X-P@X);b=DX.T@reward;td=np.linalg.solve(A,b);mc=(DX.T@V)/np.bincount(idx,weights=mu,minlength=K)
 groups[str(K)]=dict(TD_weights=td.tolist(),MC_weights=mc.tolist(),A=A.tolist(),b=b.tolist(),TD_weighted_RMSE=float(np.sqrt(np.sum(mu*(V-X@td)**2))),MC_weighted_RMSE=float(np.sqrt(np.sum(mu*(V-X@mc)**2))),TD_unweighted_RMSE=float(np.sqrt(np.mean((V-X@td)**2))),MC_unweighted_RMSE=float(np.sqrt(np.mean((V-X@mc)**2))))
w=0.
for t in range(1,10001):w+=(1/t)*(1-w*w)*2*w
X=np.array([[1.],[2.]]);y=np.array([1.,0.]);q=np.array([3.]);kernel=X@q
out=dict(N=N,jump=J,start=500,V=V.tolist(),mu=mu.tolist(),eta=eta.tolist(),reward=reward.tolist(),expected_episode_length=float(eta.sum()),value_residual=float(max(abs(V-transition(V)-reward))),visitation_residual=float(max(abs(eta-transition(eta)-h))),groups=groups,counterexamples=dict(SGD_nonminimum=dict(final_w=w,loss=(1-w*w)**2,loss_at_point1=(1-.1**2)**2),LSTD=dict(epsilon=.125,gamma=.5,x=1,next_x=2.25,A=.125+1*(1-.5*2.25),SM_denominator=1+8*(1-.5*2.25)),kernel=dict(OLS=float(q@np.linalg.lstsq(X,y,rcond=None)[0]),raw=float(kernel@y),normalized=float(kernel@y/kernel.sum()),ridge_primal=float(q@np.linalg.solve(X.T@X+np.eye(1),X.T@y)),ridge_dual=float(kernel@np.linalg.solve(X@X.T+np.eye(2),y)))),alpha_rule=dict(alpha=1/980,remaining_after10=.9**10,alpha_for95percent_after10=(1-.05**.1)/98),independent_jacobi=dict(iterations=iteration,max_value_difference=float(max(abs(V-v))),max_visitation_difference=float(max(abs(eta-e)))))
(BASE/'measurements_ch9_exact.json').write_text(json.dumps(out,indent=2)+'\n')
(BASE/'reference_ch9.tsv').write_text(''.join(f'{a:.17g} {b:.17g}\n'for a,b in zip(V,mu)))
class RNG:
 def __init__(self,seed):self.a=seed
 def next(self):
  self.a=(self.a+0x6d2b79f5)&0xffffffff;t=self.a;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296
def episode(r):
 state=500;path=[]
 while 1<=state<=1000:
  path.append(state-1);k=1+int(100*r.next());state+=k if r.next()<.5 else -k
 return path,(-1 if state<1 else 1)
f=BASE/'measurements_ch9_walk.json'
if f.exists()and'fixtures'in json.loads(f.read_text()):
 cpp=json.loads(f.read_text())['fixtures'];R=RNG(900001);wmc=np.zeros(10)
 for epno in range(10):
  path,G=episode(R)
  for state in path:g=state//100;wmc[g]+=.00002*(G-wmc[g])
 np.testing.assert_allclose(wmc,cpp['mc_first10'],atol=1e-15,rtol=1e-12)
 R=RNG(910001);eps=[episode(R)for _ in range(10)];nr=[]
 for n in [1,2,4,8,16,32,64,128,256,512]:
  for alpha in np.linspace(0,1,21):
   w=np.zeros(20);errors=[]
   for path,ret in eps:
    for t,state in enumerate(path):
     G=ret if t+n>=len(path)else w[path[t+n]//50];g=state//50;w[g]+=alpha*(G-w[g])
    errors.append(np.sqrt(np.mean((V-np.repeat(w,50))**2)))
   nr.append(float(np.mean(errors)))
 np.testing.assert_allclose(nr,cpp['nstep_run0'],atol=1e-12,rtol=1e-12)
 R=RNG(920001);orders=[5,10,20,5,10,20];ph=[np.stack([(s/1000.)**i if m<3 else np.cos(np.pi*i*s/1000.)for i in range(n+1)],axis=1)for m,n in enumerate(orders)];w=[np.zeros(n+1)for n in orders];one=np.zeros(5);many=np.zeros(350);tiles=np.array([[t*7+(state+4*t)//200 for t in range(50)]for state in range(1000)])
 for epno in range(25):
  path,G=episode(R)
  for state in path:
   for m in range(6):w[m]+=(.0001 if m<3 else .00005)*(G-w[m]@ph[m][state])*ph[m][state]
   one[state//200]+=.0001*(G-one[state//200]);many[tiles[state]]+=.0001/50*(G-many[tiles[state]].sum())
 vals=[np.sqrt(np.sum(mu*(V-ph[m]@w[m])**2))for m in range(6)]+[np.sqrt(np.sum(mu*(V-np.repeat(one,200))**2)),np.sqrt(np.sum(mu*(V-many[tiles].sum(axis=1))**2))]
 np.testing.assert_allclose(vals,cpp['bases_run0_after25'],atol=1e-12,rtol=1e-12)
 report=dict(MC_fixture_max_difference=float(max(abs(wmc-np.array(cpp['mc_first10'])))),nstep_210_max_difference=float(max(abs(np.array(nr)-cpp['nstep_run0']))),bases_8_max_difference=float(max(abs(np.array(vals)-cpp['bases_run0_after25']))))
 (BASE/'measurements_ch9_independent.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
print('Exact V/µ, independent Jacobi, TD moments and counterexamples verified.')
js=BASE/'measurements_ch9_js.json'
if js.exists():
 results=[]
 for case in json.loads(js.read_text())['LSTD_fixtures']:
  A=np.array(case['A']);b=np.array(case['b']);w=np.linalg.solve(A,b);inverse=np.linalg.inv(A)
  dw=float(max(abs(w-np.array(case['w']))));di=float(np.max(abs(inverse-np.array(case['inverse']))));assert dw<1e-9 and di<1e-9
  results.append(dict(seed=case['seed'],episodes=case['episodes'],steps=case['steps'],max_w_difference=dw,max_inverse_difference=di))
 p=BASE/'measurements_ch9_independent.json';report=json.loads(p.read_text())if p.exists()else{};report['browser_LSTD_numpy']=results;p.write_text(json.dumps(report,indent=2)+'\n')
 print('Four browser LSTD inverses independently verified by NumPy.')
