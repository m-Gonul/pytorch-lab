"""Independent NumPy references and actual Markov-stream test for Chapter6.
OPENBLAS_NUM_THREADS=1 python3 -I notes/sutton-2018/measure_ch6.py OUT.json
"""
from pathlib import Path
import numpy as np,json,sys,runpy,contextlib,io
R=Path(__file__).resolve().parents[2]

def constant_moments(alpha):
    P=np.array([[.8,.2],[.3,.7]]);r=np.array([0.,1.]);g=.9;d=np.array([.6,.4]);true=np.linalg.solve(np.eye(2)-g*P,r)
    M=np.zeros((4,4));b=np.zeros(4)
    for i in range(2):
        for j in range(2):
            A=np.eye(2);A[i,i]-=alpha;A[i,j]+=alpha*g
            M[j*2:(j+1)*2,i*2:(i+1)*2]+=P[i,j]*A
            b[j*2+i]+=alpha*P[i,j]*r[i]*d[i]
    joint=np.linalg.solve(np.eye(4)-M,b).reshape(2,2);m=(d[:,None]*true).reshape(-1)
    for k in range(100000):
        n=M@m+b
        if np.max(abs(n-m))<1e-14:break
        m=n
    assert np.max(abs(m-joint.reshape(-1)))<1e-9
    return dict(alpha=alpha,P=P,r=r,gamma=g,d=d,true=true,M=M,b=b,joint=joint,mean=joint.sum(0),bias=joint.sum(0)-true,spectral_radius=float(max(abs(np.linalg.eigvals(M)))),iterations=k+1)

def constant_simulation(N=4096,T=12000,burn=6000,alpha=.1):
    P=np.array([[.8,.2],[.3,.7]]);reward=np.array([0.,1.]);gamma=.9;d=np.array([.6,.4]);true=np.linalg.solve(np.eye(2)-gamma*P,reward)
    rng=np.random.default_rng(1);s=(rng.random(N)>=d[0]).astype(int);v=np.broadcast_to(true,(N,2)).copy();sums=np.zeros_like(v);ids=np.arange(N)
    for t in range(T):
        n=(rng.random(N)>=P[s,0]).astype(int)
        v[ids,s]+=alpha*(reward[s]+gamma*v[ids,n]-v[ids,s]);s=n
        if t>=burn:sums+=v
    means=sums/(T-burn)
    return dict(chains=N,steps_per_chain=T,burn=burn,alpha=alpha,seed=1,mean=means.mean(0),se=means.std(0,ddof=1)/np.sqrt(N))

def cliff_models(epsilon=.1):
    nexts=np.zeros((48,4),int);reward=-np.ones((48,4));moves=[(-1,0),(0,1),(1,0),(0,-1)]
    for s in range(48):
        for a,(dr,dc) in enumerate(moves):
            row=np.clip(s//12+dr,0,3);col=np.clip(s%12+dc,0,11);n=row*12+col
            if row==3 and 1<=col<=10:n=36;reward[s,a]=-100
            nexts[s,a]=n
    output={}
    for soft in [False,True]:
        V=np.zeros(48)
        for it in range(20000):
            Q=reward+V[nexts];W=(1-epsilon)*Q.max(1)+epsilon*Q.mean(1) if soft else Q.max(1);W[47]=0
            err=np.max(abs(W-V));V=W
            if err<1e-12:break
        Q=reward+V[nexts];Q[47]=0
        mask=Q>=Q.max(1,keepdims=True)-1e-9;pi=epsilon/4+(1-epsilon)*mask/mask.sum(1,keepdims=True)
        P=np.zeros((48,48));rp=(pi*reward).sum(1)
        for s in range(47):
            for a in range(4):P[s,nexts[s,a]]+=pi[s,a]
        ids=np.arange(47);behavior=np.zeros(48);behavior[ids]=np.linalg.solve(np.eye(47)-P[:47,:47],rp[:47])
        output['expected_soft_optimum' if soft else 'greedy_optimum']=dict(V=V,Q=Q,policy=pi,epsilon_behavior_V=behavior,iterations=it+1,residual=err)
    return output

def wind():
    with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path(str(R/'content/modules/m11-5-sarsa-q/x6.sol.py'))
    output={}
    for actions in [4,8,9]:
        for stochastic in [False,True]:
            V,pi=ns['optimal_windy'](actions,stochastic);output[f'{actions}/{int(stochastic)}']=dict(V=V,policy=pi)
    out=ns['sarsa_windy'](8000,4,False,.5,.1,6)
    output['python_train']=out
    return output

def arithmetic():
    elapsed=np.array([0,5,20,30,40,43]);predicted=np.array([30,35,15,10,3,0]);rewards=np.diff(elapsed);actual=43-elapsed
    P=np.zeros((5,5));reward=np.zeros(5);reward[4]=.5
    for i in range(5):
        if i:P[i,i-1]=.5
        if i<4:P[i,i+1]=.5
    true=np.linalg.solve(np.eye(5)-P,reward)
    # Exact state-version residual for6.1.
    V=0.;deltas=[];tables=[V]
    for i,r in enumerate([1.,1.]):
        delta=r+(V if i==0 else 0)-V;deltas.append(delta);V+=.1*delta;tables.append(V)
    first=np.array([1.,0.]);first[0]=-2+.9*first.max()
    return dict(driving=dict(elapsed=elapsed,predicted_remaining=predicted,predicted_total=elapsed+predicted,rewards=rewards,actual_remaining=actual,mc_error=actual-predicted,td_error=rewards+predicted[1:]-predicted[:-1],highway_alpha_half_new=15+.5*(23-15)),random_walk_true=true,ex6_1=dict(deltas=deltas,table_versions=tables,td_sum=sum(deltas),mc_error=2.,correction=tables[1]-tables[0]),ex6_3_new_A=.5+.1*(0-.5),ex6_12=dict(first_table=first,sarsa_carried_action=0,q_reselected_action=int(first.argmax())),predictor=dict(VB=6/8,VA_MC=0.,VA_TD=6/8,squared_error_MC=6*(.75-1)**2+2*.75**2,squared_error_TD=6*(.75-1)**2+3*.75**2),afterstate_cost_example=dict(post=[10,10],before=[[10,10],[11,9]],actions=[0,1],known_costs=[0,2],Q_difference=2))

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating)):return x.item()
    if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    return x
if __name__=='__main__':
    out=dict(arithmetic=arithmetic(),constant_alpha=dict(moments=[constant_moments(a)for a in [.5,.1,.01]],simulation=constant_simulation()),wind=wind(),cliff=cliff_models())
    Path(sys.argv[1]).write_text(json.dumps(serial(out),ensure_ascii=False,indent=2)+'\n')
    print('Cliff references',out['cliff']['greedy_optimum']['V'][36],out['cliff']['greedy_optimum']['epsilon_behavior_V'][36],out['cliff']['expected_soft_optimum']['V'][36])
    print('Constant-alpha true',out['constant_alpha']['moments'][1]['true'],'mean',out['constant_alpha']['moments'][1]['mean'])
