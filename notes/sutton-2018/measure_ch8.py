"""Independent Dyna and dense NumPy verification of the C++ policy evaluator.
Run measure_ch8.mjs --verify and ch8trajectory verify first (see README).
"""
from pathlib import Path
import json,runpy,contextlib,io
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path(str(ROOT/'content/modules/m11-10-planning/x5.sol.py'))
U32=ns['U32']

def dyna(n,seed):
 Q=np.zeros((54,4));model={};actions={};lengths=[];behavior=U32(seed);plan=U32(seed+100000)
 walls={(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7)};moves=[(-1,0),(1,0),(0,-1),(0,1)]
 def update(s,a,z,r,done):Q[s,a]+=.1*(r+(0 if done else .95*max(Q[z]))-Q[s,a])
 steps=0
 for ep in range(3):
  s=18;k=0
  while True:
   if behavior.random()<.1:a=behavior.index(4)
   else:
    best=max(Q[s]);choices=[a for a in range(4)if Q[s,a]==best];a=choices[behavior.index(len(choices))]
   r,c=divmod(s,9);dr,dc=moves[a];y,x=r+dr,c+dc;z=9*y+x if 0<=y<6 and 0<=x<9 and(y,x)not in walls else s
   reward=float(z==8);done=z==8;update(s,a,z,reward,done)
   if s not in actions:actions[s]=[]
   if(s,a)not in model:actions[s].append(a)
   model[s,a]=(z,reward,done)
   for _ in range(n):
    states=list(actions);ps=states[plan.index(len(states))];aa=actions[ps];pa=aa[plan.index(len(aa))];update(ps,pa,*model[ps,pa])
   s=z;steps+=1;k+=1
   if done:break
   if k>=100000:raise AssertionError('unexpected guard')
  lengths.append(k)
 return dict(Q=Q.ravel().tolist(),lengths=lengths,steps=steps,updates=(n+1)*steps)

out={'dyna':[],'trajectory':[]}
for row in json.loads((BASE/'measurements_ch8_dyna_fixtures.json').read_text()):
 py=dyna(row['n'],row['seed']);difference=float(np.max(abs(np.array(py['Q'])-row['Q'])));assert difference<1e-12
 assert py['lengths']==row['lengths'] and py['steps']==row['steps'] and py['updates']==row['updates']
 out['dyna'].append(dict(seed=row['seed'],n=row['n'],**py,max_JS_difference=difference))
for row in json.loads((BASE/'measurements_ch8_cpp_fixtures.json').read_text()):
 b=row['b'];nxt,rew=ns['make_task'](17,b,800001)
 assert np.array_equal(nxt.ravel(),row['next']) and np.max(abs(rew.ravel()-row['reward']))<1e-14
 for method,mode in enumerate(['uniform','trajectory']):
  exact_values=[];diffs=[]
  for j,steps in enumerate([0,10,50,100]):
   r=ns['learn'](nxt,rew,steps,mode,1700001,points=[steps]);Q=r['Q'];a=np.argmax(Q,axis=1);P=np.zeros((17,17));reward=.9*rew[np.arange(17),a].mean(axis=1)
   for s in range(17):
    for z in nxt[s,a[s]]:P[s,z]+=.9/b
   V=np.linalg.solve(np.eye(17)-P,reward);difference=abs(float(V[0])-row['values'][method][j]);assert difference<3e-11;diffs.append(difference);exact_values.append(float(V[0]))
  out['trajectory'].append(dict(b=b,method=mode,points=[0,10,50,100],dense_Numpy_V0=exact_values,maximum_CPP_difference=max(diffs)))
out['arithmetic']=dict(uniform_outlier_variance=.01*.99,skewed_outlier_variance=.5*.5,rtdp_policy_reduction_percent=(136725-127600)/136725*100)
(BASE/'measurements_ch8_reference.json').write_text(json.dumps(out,indent=2)+'\n');print('6 Dyna full-Q fixtures and16 dense policy values verified',max(r['maximum_CPP_difference']for r in out['trajectory']))
