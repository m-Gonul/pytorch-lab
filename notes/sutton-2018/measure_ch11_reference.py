"""Own linear algebra and expected-update references for chapter11, source data never copied."""
from pathlib import Path
import numpy as np,json,math
BASE=Path(__file__).resolve().parent
OUT=BASE/'measurements_ch11_reference.json'
X=np.zeros((7,8));X[np.arange(6),np.arange(6)]=2;X[:6,7]=1;X[6,6:]=[1,2];D=np.eye(7)/7;P=np.zeros((7,7));P[:,6]=1;gamma=.99;w0=np.r_[np.ones(6),10,1]
C=X.T@D@X;Pi=X@np.linalg.pinv(C)@X.T@D;assert max(abs(Pi-np.eye(7)).ravel())<1e-12
A=X.T@D@(np.eye(7)-gamma*P)@X
profiles={}
for mode,alpha,beta in [('DP',.01,.05),('TDC',.005,.05),('GTD2',.005,.05),('ETD',.03,0)]:
 w=w0.copy();u=np.zeros(8);m=np.zeros(7);hist=[]
 for t in range(1001):
  v=X@w;delta=gamma*P@v-v;hist.append(dict(t=t,w=w.tolist(),aux=u.tolist(),RMSVE=float(np.sqrt(v@D@v)),RMSPBE=float(np.sqrt(delta@D@delta))))
  if t==1000:break
  if mode=='DP':dw=alpha*X.T@D@delta
  elif mode in ['TDC','GTD2']:
   z=X@u
   dw=alpha*((X-gamma*P@X).T@D@z if mode=='GTD2'else X.T@D@delta-gamma*(P@X).T@D@z)
   u+=beta*X.T@D@(delta-z)
  else:
   m=np.ones(7)/7+gamma*P.T@m;dw=alpha*X.T@(m*delta)
  w+=dw
 profiles[mode]=dict(alpha=alpha,beta=beta,history=hist)
# Analytic expected TD linear map eigenvalues and a null-space-preserving optimum.
null=np.r_[np.ones(6),4,-2];assert max(abs(X@null))==0
baird=dict(X=X.tolist(),P_target=P.tolist(),mu_behavior=np.diag(D).tolist(),gamma=gamma,w0=w0.tolist(),Pi=Pi.tolist(),A=A.tolist(),A_eigenvalues=[dict(real=float(v.real),imag=float(v.imag))for v in np.linalg.eigvals(A)],null_vector=null.tolist(),same_null_projection_optimum=(null*(null@w0)/(null@null)).tolist(),profiles=profiles)
# Four objectives in an explicitly chosen3-state example.
P3=np.array([[.1,.6,.3],[.2,.5,.3],[.3,.3,.4]]);r=np.array([1.,-1.,.5]);Z=np.array([[1.,0.],[1.,1.],[0.,1.]]);g=.9
B=P3.T-np.eye(3);B[-1]=1;mu=np.linalg.solve(B,np.r_[0.,0.,1.]);d=np.diag(mu);v=np.linalg.solve(np.eye(3)-g*P3,r);N=(np.eye(3)-g*P3)@Z;C3=Z.T@d@Z;PI=Z@np.linalg.inv(C3)@Z.T@d
M=[];tar=[]
for i in range(3):
 for j in range(3):
  scale=math.sqrt(mu[i]*P3[i,j]);M.append(scale*(Z[i]-g*Z[j]));tar.append(scale*r[i])
M=np.array(M);tar=np.array(tar);weights=dict(VE=np.linalg.solve(C3,Z.T@d@v),BE=np.linalg.solve(N.T@d@N,N.T@d@r),PBE=np.linalg.solve(Z.T@d@N,Z.T@d@r),TDE=np.linalg.lstsq(M,tar,rcond=None)[0]);second=np.linalg.solve(np.eye(3)-g*g*P3,r*r+2*g*r*(P3@v));var=second-v*v
metrics={}
for name,w in weights.items():
 e=r-N@w;metrics[name]=dict(w=w.tolist(),values=(Z@w).tolist(),VE=float((Z@w-v)@d@(Z@w-v)),BE=float(e@d@e),PBE=float((PI@e)@d@(PI@e)),TDE=float(np.sum((M@w-tar)**2)),RE=float((Z@w-v)@d@(Z@w-v)+mu@var))
geometry=dict(P=P3.tolist(),r=r.tolist(),X=Z.tolist(),gamma=g,mu=mu.tolist(),true_value=v.tolist(),return_variance=var.tolist(),Pi=PI.tolist(),metrics=metrics,initial_w=[0.,0.],Bellman_iterates=[],projected_iterates=[])
a=np.zeros(3);b=np.zeros(3)
for t in range(30):geometry['Bellman_iterates'].append(a.tolist());geometry['projected_iterates'].append(b.tolist());a=r+g*P3@a;b=PI@(r+g*P3@b)
# Exact A-split/presplit feature objectives, with terminal as a zero row.
def episodic(P,r,X,mu):
 n=len(r);d=np.diag(mu);N=(np.eye(n)-P)@X;v=np.linalg.solve(np.eye(n)-P,r);C=X.T@d@X;proj=X@np.linalg.pinv(C)@X.T@d;M=[];tar=[]
 for i in range(n):
  for j in range(n+1):
   p=P[i,j]if j<n else 1-sum(P[i]);
   if p:
    scale=math.sqrt(mu[i]*p);M.append(scale*(X[i]-(X[j]if j<n else 0)));tar.append(scale*r[i])
 M=np.array(M);tar=np.array(tar);out={}
 for k,w in dict(VE=np.linalg.lstsq(np.sqrt(d)@X,np.sqrt(d)@v,rcond=None)[0],BE=np.linalg.lstsq(np.sqrt(d)@N,np.sqrt(d)@r,rcond=None)[0],TDE=np.linalg.lstsq(M,tar,rcond=None)[0]).items():
  e=r-N@w;out[k]=dict(w=w.tolist(),values=(X@w).tolist(),VE=float((X@w-v)@d@(X@w-v)),BE=float(e@d@e),TDE=float(sum((M@w-tar)**2)))
 return dict(P=P.tolist(),r=r.tolist(),X=X.tolist(),mu=mu,true_value=v.tolist(),metrics=out)
split=episodic(np.array([[0,.5,.5],[0,0,0],[0,0,0]]),np.array([0.,1.,0.]),np.eye(3),[.5,.25,.25])
presplit=episodic(np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]),np.array([0.,0.,1.,0.]),np.array([[1.,0,0],[1,0,0],[0,1,0],[0,0,1]]),[.25]*4)
assert np.max(abs(np.array(split['metrics']['TDE']['w'])-[.5,.75,.25]))<1e-12
assert np.max(abs(np.array(presplit['metrics']['BE']['w'])-[.5,.75,.25]))<1e-12
# SourceBEpair exact rational functions verified by symbolic normal equations.
BEcases=[]
for g in[0,.1,.5,.9,.99,.999,.9999]:
 p=np.array([[0,.5,.5],[1,0,0],[0,.5,.5]]);x=np.array([[1.,0],[0,1],[0,1]]);r=np.array([0.,1.,-1.]);N=(np.eye(3)-g*p)@x;w=np.array([-g/(1+g+g*g),g/(2*(1+g+g*g))]);res=N.T@(N@w-r);assert max(abs(res))<1e-12
 BEcases.append(dict(gamma=g,w_raw=w.tolist(),w_B0=(w-w[1]).tolist(),BE=float(np.mean((r-N@w)**2)),normal_residual=max(abs(res))))
OUT.write_text(json.dumps(dict(baird=baird,geometry=geometry,A_split=split,A_presplit=presplit,BE_unlearnable_pair=BEcases,no_PBE_zero=dict(gamma=.8,mu=[4/7,3/7],X=[[1],[2]],C=16/7,A=0,b=10/7,PBE=25/28),offpolicy_variance_counterexample=dict(target=[.5,.5],behavior=[.1,.9],rewards=[0,2],target_mean=1,target_variance=1,IS_mean=1,IS_variance=1/9),local_linear_not_averager=dict(training_x=[0,1],targets=[0,1],query=2,prediction=2)),indent=2)+'\n')
print('Baird profiles',[(k,v['history'][-1]['RMSVE'],v['history'][-1]['RMSPBE'])for k,v in profiles.items()]);print('four objective points', {k:v['w']for k,v in metrics.items()});print('exact references saved',OUT)
