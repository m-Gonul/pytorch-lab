import numpy as np

WIND = np.array([0,0,0,1,1,1,2,2,1,0])
MOVES = [(-1,0),(0,1),(1,0),(0,-1),(-1,-1),(-1,1),(1,1),(1,-1),(0,0)]
START, GOAL = 30, 37


def windy_step(s,a,stochastic=False,rng=None):
    row,col=divmod(s,10)
    noise=int(rng.integers(-1,2)) if stochastic and WIND[col] else 0
    dr,dc=MOVES[a]
    return int(np.clip(row+dr-WIND[col]-noise,0,6))*10+int(np.clip(col+dc,0,9))


def optimal_windy(actions=8,stochastic=False):
    P={}
    for s in range(70):
        row,col=divmod(s,10)
        for a in range(actions):
            noise=(-1,0,1) if stochastic and WIND[col] else (0,)
            dr,dc=MOVES[a]
            P[s,a]=[(int(np.clip(row+dr-WIND[col]-z,0,6))*10+int(np.clip(col+dc,0,9)),1/len(noise)) for z in noise]
    V=np.zeros(70);policy=np.zeros(70,dtype=int)
    for _ in range(10000):
        W=np.zeros(70)
        for s in range(70):
            if s==GOAL:continue
            q=[-1+sum(p*V[n] for n,p in P[s,a]) for a in range(actions)]
            policy[s]=int(np.argmax(q));W[s]=max(q)
        err=np.max(abs(W-V));V=W
        if err<1e-12:break
    else:raise RuntimeError('Model yakınsamadı')
    return V,policy


def sarsa_windy(steps=8000,actions=8,stochastic=False,alpha=.5,epsilon=.1,seed=1):
    rng=np.random.default_rng(seed);Q=np.zeros((70,actions));counts=np.zeros_like(Q,dtype=int)
    def choose(s):
        if rng.random()<epsilon:return int(rng.integers(actions))
        best=np.flatnonzero(Q[s]==Q[s].max());return int(rng.choice(best))
    s=START;a=choose(s);episodes=0;unfinished=0;lengths=[];history=[]
    for _ in range(steps):
        n=windy_step(s,a,stochastic,rng);done=n==GOAL
        a2=0 if done else choose(n) # SARSA seçimi Q güncellemesinden önce.
        counts[s,a]+=1
        Q[s,a]+=alpha*(-1+(0 if done else Q[n,a2])-Q[s,a]);unfinished+=1
        if done:
            episodes+=1;lengths.append(unfinished);unfinished=0;s=START;a=choose(s)
        else:s,a=n,a2
        history.append(episodes)
    return dict(Q=Q,counts=counts,history=history,lengths=lengths,unfinished=unfinished)


def trajectory(policy,stochastic=False,seed=1,budget=1000):
    rng=np.random.default_rng(seed);s=START;path=[s]
    for _ in range(budget):
        s=windy_step(s,int(policy[s]),stochastic,rng);path.append(s)
        if s==GOAL:return path,True
    return path,False
