"""Independent Python replay and whitened least-squares/gradient checks for chapter11."""
from pathlib import Path
import json,math
import numpy as np
from scipy.linalg import lstsq
BASE=Path(__file__).resolve().parent
S=json.loads((BASE/'measurements_ch11_samples.json').read_text());E=json.loads((BASE/'measurements_ch11_reference.json').read_text())
class RNG:
 def __init__(self,seed):self.state=seed
 def next(self):
  self.state=(self.state+0x6d2b79f5)&0xffffffff;t=self.state;t=((t^(t>>15))*(t|1))&0xffffffff;t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff;return((t^(t>>14))&0xffffffff)/4294967296

def feature(s):
 v=[0.]*8
 if s<6:v[s]=2;v[7]=1
 else:v[6]=1;v[7]=2
 return v

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def q(w,s):return 2*w[s]+w[7]if s<6 else w[6]+2*w[7]
max_rel=0.;max_qrel=0.;fixtures=0
for case in S['samples']:
 rng=RNG(case['seed']);state=int(rng.next()*7);w=[[1.]*6+[10.,1.]for _ in range(4)];u=[[0.]*8 for _ in range(4)];Q=[[1.]*6+[10.,1.]for _ in range(2)];follow=0.;prev_rho=0.;visits=[0]*14
 for t in range(100000):
  solid=rng.next()>=6/7;a=int(solid);new=6 if solid else int(rng.next()*6);rho=7 if solid else 0;visits[2*state+a]+=1
  if t<1000:
   x=feature(state);xp=feature(new);follow=1+.99*prev_rho*follow
   for k in range(4):
    delta=.99*dot(w[k],xp)-dot(w[k],x);vx=dot(u[k],x);alpha=[.01,.005,.005,.03][k];oldw=w[k][:];oldu=u[k][:]
    for i in range(8):
     if k==0:dw=alpha*rho*delta*x[i]
     elif k==1:dw=alpha*rho*(delta*x[i]-.99*xp[i]*vx)
     elif k==2:dw=alpha*rho*(x[i]-.99*xp[i])*vx
     else:dw=alpha*follow*rho*delta*x[i]
     w[k][i]=oldw[i]+dw
     if k in[1,2]:u[k][i]=oldu[i]+.05*rho*(delta-vx)*x[i]
  delta=.99*max(q(Q[0],new),q(Q[1],new))-q(Q[a],state);idx=state if state<6 else 6;c1=2 if state<6 else 1;c2=1 if state<6 else 2
  Q[a][idx]+=(.01*delta)*c1;Q[a][7]+=(.01*delta)*c2
  if t<100:
   ref=case['first100'][t];assert [state,a,new]==ref[:3];assert math.isclose(delta,ref[3],rel_tol=1e-12,abs_tol=1e-12);fixtures+=1
  state=new;prev_rho=rho
 assert visits==case['Q_visits']
 for k in range(4):
  for found,expected in zip(w[k]+u[k],case['final'][k]['weights']+case['final'][k]['aux']):
   err=abs(found-expected)/max(1,abs(expected));max_rel=max(max_rel,err);assert err<1e-10,(case['seed'],k,found,expected)
 for a in range(2):
  for found,expected in zip(Q[a],case['Q_weights'][a]):
   err=abs(found-expected)/max(1,abs(expected));max_qrel=max(max_qrel,err);assert err<1e-10,(found,expected)
# Independently use whitened QR least squares instead of normal-equation solves.
g=E['geometry'];X=np.array(g['X']);P=np.array(g['P']);r=np.array(g['r']);mu=np.array(g['mu']);gamma=g['gamma'];v=np.array(g['true_value']);W=np.diag(np.sqrt(mu));D=np.diag(mu);N=(np.eye(3)-gamma*P)@X
qr_diffs=[]
for name,A,b in [('VE',W@X,W@v),('BE',W@N,W@r)]:
 w=lstsq(A,b,lapack_driver='gelsy')[0];ref=np.array(g['metrics'][name]['w']);qr_diffs.append(float(max(abs(w-ref))));assert max(abs(w-ref))<1e-11
# Projection algebra and MSPBE gradient independently finite-differenced.
C=X.T@D@X;PI=X@np.linalg.inv(C)@X.T@D;assert np.max(abs(PI@PI-PI))<1e-12;assert np.max(abs(PI.T@D-D@PI))<1e-12
A=X.T@D@N;b=X.T@D@r
value=lambda w:float((b-A@w)@np.linalg.solve(C,b-A@w))
grad_diffs=[]
for w in[np.array([0.,0.]),np.array([.4,-.7]),np.array(g['metrics']['BE']['w'])]:
 grad=-2*A.T@np.linalg.solve(C,b-A@w);numeric=np.array([(value(w+np.eye(2)[j]*1e-6)-value(w-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)]);diff=float(max(abs(grad-numeric)));grad_diffs.append(diff);assert diff<1e-8
# Source return-error proof: conditional mean residual exactly0 makes cross term vanish.
return_var=np.array(g['return_variance']);assert min(return_var)>0
for z in g['metrics'].values():assert abs(z['RE']-z['VE']-float(mu@return_var))<1e-12
out=dict(CPP_first100_transitions_checked=fixtures,CPP_4_algorithms_final_relative_difference=max_rel,CPP_Q_4x100000_final_relative_difference=max_qrel,whitened_QR_vs_normal_equations_max_differences=qr_diffs,MSPBE_gradient_finite_difference_max_differences=grad_diffs,projection_identities=True,return_error_identity=True)
(BASE/'measurements_ch11_independent.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
