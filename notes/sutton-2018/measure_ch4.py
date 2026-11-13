from pathlib import Path
import numpy as np,math,json

def location(req,ret,N=20):
    def capped(lam,n):
        p=np.array([math.exp(-lam)*lam**k/math.factorial(k) for k in range(n+1)])
        p[-1]=max(0.,1-p[:-1].sum());return p
    T=np.zeros((N+1,N+1));r=np.zeros(N+1);pr=capped(ret,N)
    for n in range(N+1):
        pq=capped(req,n);r[n]=10*np.dot(pq,np.arange(n+1))
        for k,p in enumerate(pq):np.add.at(T[n],np.minimum(n-k+np.arange(N+1),N),p*pr)
    return T,r

def jack_modified(gamma=.9,modified=True):
    if not 0<=gamma<1:raise ValueError('0≤gamma<1 gerekli')
    T1,r1=location(3,3);T2,r2=location(4,2)
    n1,n2=np.indices((21,21));acts=np.arange(-5,6);cache=[]
    for a in acts:
        ok=(a<=n1)&(-a<=n2);m1=np.clip(n1-a,0,20);m2=np.clip(n2+a,0,20)
        cost=2*(max(a-1,0) if modified and a>0 else abs(a))
        parking=4*((m1>10).astype(int)+(m2>10).astype(int)) if modified else 0
        cache.append((ok,m1,m2,r1[m1]+r2[m2]-cost-parking))
    pi=np.zeros((21,21),int);V=np.zeros((21,21));history=[]
    for iteration in range(30):
        ids=pi+5
        ok=np.stack([x[0] for x in cache]);m1=np.stack([x[1] for x in cache]);m2=np.stack([x[2] for x in cache]);R=np.stack([x[3] for x in cache])
        x=np.take_along_axis(m1,ids[None],0)[0];y=np.take_along_axis(m2,ids[None],0)[0];rp=np.take_along_axis(R,ids[None],0)[0]
        for sweep in range(10000):
            F=T1@V@T2.T;W=rp+gamma*F[x,y];delta=np.max(abs(W-V));V=W
            if delta<1e-11:break
        else:raise RuntimeError('değerlendirme yakınsamadı')
        history.append(dict(policy=pi.copy(),V=V.copy(),sweeps=sweep+1))
        F=T1@V@T2.T;Q=np.stack([np.where(valid,r+gamma*F[x,y],-np.inf) for valid,x,y,r in cache])
        best=Q.max(0);old=np.take_along_axis(Q,ids[None],0)[0]
        new=np.where(old>=best-1e-9,pi,acts[Q.argmax(0)])
        if np.array_equal(new,pi):return dict(V=V,policy=pi,history=history,residual=float(np.max(abs(best-V))))
        pi=new
    raise RuntimeError('politika sonlanmadı')

def grid_values(steps=None,changed=False):
    # Original visual terminal at16index15; new source state15 is stored as index16.
    n=17 if steps=='extra' else 16;P=np.zeros((n,n));R=-np.ones(n);R[[0,15]]=0
    for s in range(1,15):
        row,col=divmod(s,4)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc=row+dr,col+dc;t=s if not(0<=rr<4 and 0<=cc<4) else rr*4+cc
            if changed and s==13 and dr==1:t=16
            P[s,t]+=.25
    if n==17:
        for t in [12,13,14,16]:P[16,t]+=.25
    ids=[s for s in range(n) if s not in [0,15]];V=np.zeros(n)
    if steps is None or steps=='extra':V[ids]=np.linalg.solve(np.eye(len(ids))-P[np.ix_(ids,ids)],R[ids])
    else:
        for _ in range(steps):V=R+P@V
    return V

def gambler_trace(ph=.4,eps=1e-12):
    V=np.zeros(101);V[100]=1;trace={}
    for k in range(1,30000):
        old=V.copy()
        for s in range(1,100):
            a=np.arange(1,min(s,100-s)+1);V[s]=(ph*V[s+a]+(1-ph)*V[s-a]).max()
        if k in [1,2,3,32]:trace[str(k)]=V.copy()
        if k>=32 and np.max(abs(V-old))<eps:break
    else:raise RuntimeError('gambler yakınsamadı')
    pi=np.zeros(101,int)
    for s in range(1,100):
        a=np.arange(1,min(s,100-s)+1);Q=ph*V[s+a]+(1-ph)*V[s-a];pi[s]=a[np.argmax(np.round(Q,9))]
    return dict(V=V,policy=pi,trace=trace,sweeps=k)

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,list):return [serial(v) for v in x]
    return x
if __name__=='__main__':
    out={'grid':{str(k):grid_values(k) for k in [0,1,2,3,10]},'grid_exact':grid_values(),'ex4_2_unchanged':grid_values('extra')[-1],'ex4_2_changed':grid_values('extra',True)[-1], 'jack':jack_modified(modified=False),'jack_modified':jack_modified(),'gambler':{str(p):gambler_trace(p) for p in [.4,.25,.55]},'thousand_year_lower_bound':1e20/1e6/(365.25*86400)}
    out['gambler_tolerance']={str(p):{str(eps):gambler_trace(p,eps) for eps in [1e-6,1e-9,1e-12]} for p in [.25,.55]}
    r=.45/.55;out['ph55_analytic_max_error']=float(max(abs(v-(1-r**s)/(1-r**100)) for s,v in enumerate(out['gambler']['0.55']['V'])))
    out['optimal_family_examples']={}
    V=out['gambler']['0.4']['V']
    for s in [25,26,50,51]:
        Q={a:.4*V[s+a]+.6*V[s-a] for a in range(1,min(s,100-s)+1)};m=max(Q.values());out['optimal_family_examples'][str(s)]=[a for a,v in Q.items() if abs(v-m)<1e-12]
    import runpy,contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):
        ns=runpy.run_path(str(Path(__file__).resolve().parents[2]/'content/modules/m11-3-policy-iteration/x3.sol.py'))
    out['x3_synchronous']={}
    for ph in [.25,.55]:
        rows={}
        for eps in [1e-6,1e-9,1e-12]:
            v,pi=ns['gambler_vi'](ph,eps);rows[str(eps)]=dict(V=v,policy=pi)
        out['x3_synchronous'][str(ph)]=rows
    Path(__import__('sys').argv[1]).write_text(json.dumps(serial(out),ensure_ascii=False,indent=2)+'\n')
    print('extra',out['ex4_2_unchanged'],out['ex4_2_changed'])
    for key in ['jack','jack_modified']:
        z=out[key];print(key,'history',len(z['history']),'V',float(z['V'].min()),float(z['V'].max()),'policy(20,0)',z['policy'][20,0],'residual',z['residual'])
    for key,z in out['gambler'].items():print('gambler',key,'sweeps',z['sweeps'],'V50',z['V'][50],'pi50/51',z['policy'][50:52])
