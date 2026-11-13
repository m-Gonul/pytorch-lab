import numpy as np
import math

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



result=jack_modified()
print(result["policy"][20,0], round(float(result["V"][0,0]),3))
