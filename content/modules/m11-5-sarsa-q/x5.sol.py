import numpy as np


def prefix_targets(rewards, ratios, gamma):
    """At each t, return unnormalized discount-aware Q numerator and weight.

    ratios[k]=pi(A_k|S_k)/b(A_k|S_k), rewards[k]=R_{k+1}.
    The current forced action ratio is excluded from Q's suffix.
    """
    r=np.asarray(rewards,dtype=float); w=np.asarray(ratios,dtype=float)
    T=len(r); N=np.zeros(T); D=np.zeros(T)
    for t in range(T):
        z,flat=1.,0.
        for h in range(t+1,T+1):
            flat+=r[h-1]
            c=gamma**(h-t-1)*(1-gamma if h<T else 1.)
            N[t]+=c*z*flat; D[t]+=c*z
            if h<T: z*=w[h]
            if z==0: break
    return N,D


def update_episode(Q,C,states,actions,rewards,behavior,gamma):
    """Update the greedy target after each reverse-time Q update; no outer break."""
    Q=np.array(Q,dtype=float,copy=True); C=np.array(C,dtype=float,copy=True)
    T=len(rewards)
    for t in reversed(range(T)):
        # Recompute the current target for this t; behavior is the saved probability.
        policy=Q.argmax(axis=1)
        ratios=[float(actions[k]==policy[states[k]])/behavior[k] for k in range(T)]
        N,D=prefix_targets(rewards[t:],ratios[t:],gamma)
        s,a=states[t],actions[t]
        if D[0]>0:
            C[s,a]+=D[0]
            Q[s,a]+=(N[0]-D[0]*Q[s,a])/C[s,a]
    return Q,C
