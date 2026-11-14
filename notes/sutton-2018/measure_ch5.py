"""Independent NumPy Blackjack and complete source-track references; no source images.
Run with OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch5.py OUT.json.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np,random,json,runpy,contextlib,io,sys
R=Path(__file__).resolve().parents[2]
P=[1/13]*9+[4/13]
def add(t,u,c):
    if c==1:
        if t+11<=21:t+=11;u=True
        else:t+=1
    else:t+=c
    if t>21 and u:t-=10;u=False
    return t,u
@lru_cache(None)
def dealer(t,u):
    out=np.zeros(6)
    if t>=17:out[min(t-17,5)]=1;return out
    for c,p in enumerate(P,1):out+=p*dealer(*add(t,u,c))
    return out
D=np.array([sum((p*dealer(*add(11 if d==1 else d,d==1,c)) for c,p in enumerate(P,1)),np.zeros(6)) for d in range(1,11)])
def idx(t,u,d):return int(u)*100+(t-12)*10+d-1

def blackjack():
    stand=np.array([sum(p*(1 if j==5 or j+17<t else -1 if j+17>t else 0) for j,p in enumerate(D[d-1])) for u in range(2) for t in range(12,22) for d in range(1,11)])
    T=np.zeros((200,200));bust=np.zeros(200)
    for u in range(2):
        for t in range(12,22):
            for d in range(1,11):
                s=idx(t,u,d)
                for c,p in enumerate(P,1):
                    n,v=add(t,u,c)
                    if n>21:bust[s]+=p
                    else:T[s,idx(n,v,d)]+=p
    policy=np.array([int(t<20) for u in range(2) for t in range(12,22) for d in range(1,11)])
    V=np.linalg.solve(np.eye(200)-policy[:,None]*T,np.where(policy,-bust,stand));Vo=np.zeros(200)
    for it in range(100):
        Q=np.stack([stand,-bust+T@Vo],1);W=Q.max(1);err=np.max(abs(W-Vo));Vo=W
        if err<1e-14:break
    Q=np.stack([stand,-bust+T@Vo],1);opt=(Q[:,1]>Q[:,0]).astype(int)
    W=np.zeros((200,3));Pp=policy[:,None]*T
    for u in range(2):
        for t in range(12,22):
            for d in range(1,11):
                s=idx(t,u,d)
                if policy[s]:W[s,0]=bust[s]
                else:
                    for j,p in enumerate(D[d-1]):
                        outcome=1 if j==5 or j+17<t else -1 if j+17>t else 0;W[s,outcome+1]+=p
    outcomes=np.linalg.solve(np.eye(200)-Pp,W);assert np.max(abs(outcomes.sum(1)-1))<1e-12
    out=dict(dealer=D.tolist(),prediction=V.tolist(),optimal=Vo.tolist(),optimal_Q=Q.tolist(),optimal_policy=opt.tolist(),target_start_value=float(V[idx(13,1,2)]),start_outcomes=outcomes[idx(13,1,2)].tolist())
    rng=np.random.default_rng(1);count=rng.multinomial(100_000_000,outcomes[idx(13,1,2)])
    out['target_100million']=dict(counts=count.tolist(),value=float((count[2]-count[0])/100_000_000),method='Complete-return multinomial counts, equivalent distribution; not100million path traversals.',seed=1)
    with contextlib.redirect_stdout(io.StringIO()):
        pred=runpy.run_path(str(R/'content/modules/m11-4-mc-td/x1.sol.py'));control=runpy.run_path(str(R/'content/modules/m11-5-sarsa-q/x3.sol.py'))
    for n in [10000,500000]:
        z=pred['mc_blackjack'](n,1,return_counts=True);out[f'prediction_{n}']={k:v.tolist() for k,v in z.items()};out[f'prediction_{n}']['rmse']=float(np.sqrt(np.mean((z['V'].reshape(-1)-V)**2)))
    q,pi=control['mc_es'](1000000,1);out['control']=dict(episodes=1000000,seed=1,Q=q.tolist(),policy=pi.tolist(),value=q.max(-1).tolist(),agreement=float((pi.reshape(-1)==opt).mean()),rmse=float(np.sqrt(np.mean((q.max(-1).reshape(-1)-Vo)**2))))
    return out


def racetrack(which):
    with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path(str(R/'content/modules/m11-5-sarsa-q/x4.sol.py'))
    m=ns['track'](which);states=[(x,y,vx,vy) for y in range(m['H']) for x in range(m['W']) if (x,y) in m['road']-m['finish'] for vx in range(5) for vy in range(5) if vx or vy or (x,y,0,0) in m['starts']];index={s:i for i,s in enumerate(states)}
    S=len(states);start=np.array([index[s] for s in m['starts']]);nexts=np.full((S,9),S+1,dtype=int);valid=np.zeros((S,9),bool);noisy=np.empty(S,int)
    def encode(z,event):return S if event=='finish' else S+1 if event=='crash' else index[z]
    for i,s in enumerate(states):
        z,e=ns['crossing'](m,*s);noisy[i]=encode(z,e)
        for a in ns['actions'](m,s):
            valid[i,a]=True;dx,dy=ns['ACCEL'][a];z,e=ns['crossing'](m,*s[:2],s[2]+dx,s[3]+dy);nexts[i,a]=encode(z,e)
    V=np.zeros(S+2)
    for it in range(10000):
        V[S+1]=V[start].mean();Q=np.where(valid,-1+.9*V[nexts]+.1*V[noisy,None],-np.inf);W=Q.max(1);delta=np.max(abs(W-V[:S]));V[:S]=W
        if delta<1e-10:break
    assert delta<1e-8
    flat=[(y*m['W']+x)*25+vx*5+vy for x,y,vx,vy in states]
    # Stable NumPy ties use action order. Value equality is the cross-language invariant.
    return dict(width=m['W'],height=m['H'],road_cells=len(m['road']),states=S,flat_state_ids=flat,V=V[:S].tolist(),optimal_policy=Q.argmax(1).tolist(),start_value=float(V[start].mean()),residual=float(delta),iterations=it+1,all_legal_actions=True)
if __name__=='__main__':
    out=dict(blackjack=blackjack(),racetrack=[racetrack(k) for k in range(2)],ex5_5=dict(first=10,every=sum(range(1,11))/10),loop_model=dict(first_moment=1,second_moment='infinite',positive_length_probability='.05*.45**(L-1)',scaled_return='2**L'),single_weight_zero_counterexample=dict(weighted_expectation=.5,behavior_value=1),conditional_future_ratios=dict(given_reward0=[2,0],given_reward1=[4,0],conditional_means=[1,1]))
    Path(sys.argv[1]).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('NumPy references',out['blackjack']['target_start_value'],[x['start_value'] for x in out['racetrack']])
