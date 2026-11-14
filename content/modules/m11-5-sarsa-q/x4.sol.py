import math
import numpy as np

TRACKS = [['###.............F', '##..............F', '##..............F', '#...............F', '................F', '................F', '..........#######', '.........########', '.........########', '.........########', '.........########', '.........########', '.........########', '.........########', '#........########', '#........########', '#........########', '#........########', '#........########', '#........########', '#........########', '#........########', '##.......########', '##.......########', '##.......########', '##.......########', '##.......########', '##.......########', '##.......########', '###......########', '###......########', '###SSSSSS########'], ['################...............F', '#############..................F', '############...................F', '###########....................F', '###########....................F', '###########....................F', '###########....................F', '############...................F', '#############..................F', '##############................##', '##############.............#####', '##############............######', '##############..........########', '##############.........#########', '#############..........#########', '############...........#########', '###########............#########', '##########.............#########', '#########..............#########', '########...............#########', '#######................#########', '######.................#########', '#####..................#########', '####...................#########', '###....................#########', '##.....................#########', '#......................#########', '.......................#########', '.......................#########', 'SSSSSSSSSSSSSSSSSSSSSSS#########']]
ACCEL = [(x,y) for x in (-1,0,1) for y in (-1,0,1)]


def track(which=0):
    rows = TRACKS[which]
    H, W = len(rows), len(rows[0])
    road = {(x,H-1-j) for j,row in enumerate(rows) for x,c in enumerate(row) if c != '#'}
    starts = sorted((x,y,0,0) for j,row in enumerate(rows) for x,c in enumerate(row) if c == 'S' for y in [H-1-j])
    finish = {(x,H-1-j) for j,row in enumerate(rows) for x,c in enumerate(row) if c == 'F'}
    return dict(W=W,H=H,road=road,starts=starts,finish=finish)


def actions(m, s):
    x,y,vx,vy = s
    return [a for a,(dx,dy) in enumerate(ACCEL)
            if 0 <= vx+dx < 5 and 0 <= vy+dy < 5
            and (vx+dx or vy+dy or (x,y,0,0) in m['starts'])]


def crossing(m, x, y, vx, vy):
    # Segment/grid supercover: corner contact with either wall is a crash.
    if vx == vy == 0:
        return (x,y,vx,vy), 'road'
    xf = next(iter(m['finish']))[0]
    lo,hi = min(p[1] for p in m['finish'])-.5, max(p[1] for p in m['finish'])+.5
    tf = (xf-x)/vx if vx else math.inf
    if not (0 < tf <= 1 and lo <= y+tf*vy <= hi):
        tf = math.inf
    cx,cy,tx,ty = x,y,.5/vx if vx else math.inf,.5/vy if vy else math.inf
    while True:
        t = min(tx,ty)
        if tf <= min(t,1):
            return None, 'finish'
        if t > 1:
            return (x+vx,y+vy,vx,vy), 'road'
        if abs(tx-ty) < 1e-12:
            if not {(cx+1,cy),(cx,cy+1),(cx+1,cy+1)} <= m['road']:
                return None, 'crash'
            cx,cy,tx,ty = cx+1,cy+1,tx+1/vx,ty+1/vy
        elif tx < ty:
            cx,tx = cx+1,tx+1/vx
            if (cx,cy) not in m['road']: return None, 'crash'
        else:
            cy,ty = cy+1,ty+1/vy
            if (cx,cy) not in m['road']: return None, 'crash'


def guide(m, s):
    # Proper low-speed prior; not the solution of the stochastic full MDP.
    if 'distance' not in m:
        d = {}
        for y in reversed(range(m['H'])):
            for x in reversed(range(m['W'])):
                if (x,y) in m['finish']: d[x,y] = 0
                elif (x,y) in m['road']: d[x,y] = 1+min(d.get((x+1,y),1e6),d.get((x,y+1),1e6))
        m['distance'] = d
    scores=[]
    for a in actions(m,s):
        vx,vy=s[2]+ACCEL[a][0],s[3]+ACCEL[a][1]
        if vx == vy == 0: continue
        z,status=crossing(m,*s[:2],vx,vy)
        score=-1e6 if status=='finish' else 1e6 if status=='crash' else 1000*(max(0,vx-1)+max(0,vy-1))+m['distance'][z[:2]]
        scores.append((score,a))
    return min(scores)[1]


def step(m, s, a, rng, noise=.1):
    if a not in actions(m,s):
        raise ValueError('Geçersiz ivme')
    dx,dy = (0,0) if rng.random() < noise else ACCEL[a]
    z,status = crossing(m,*s[:2],s[2]+dx,s[3]+dy)
    if status == 'crash':
        z = m['starts'][int(rng.integers(len(m['starts'])))]
    return z,-1,status=='finish',status


def control(episodes=1000, which=0, seed=1, epsilon=.1, noise=.1):
    m,rng=track(which),np.random.default_rng(seed)
    Q,C,policy,returns={},{},{},[]
    def values(s):
        if s not in Q:
            Q[s]=np.full(9,-1000.0); C[s]=np.full(9,-np.inf); policy[s]=guide(m,s)
        return Q[s]
    def greedy(s):
        q=values(s); acts=[a for a in actions(m,s) if s[2]+ACCEL[a][0] or s[3]+ACCEL[a][1]]
        # Stable tie rule begins with the proper guide.
        order=[guide(m,s)]+[a for a in acts if a!=guide(m,s)]
        return max(order,key=lambda a:q[a])
    for ep in range(episodes):
        s=m['starts'][int(rng.integers(len(m['starts'])))]; history=[]
        while True:
            values(s); acts=actions(m,s); target=policy[s]
            a=acts[int(rng.integers(len(acts)))] if rng.random()<epsilon else target
            prob=epsilon/len(acts)+(1-epsilon if a==target else 0)
            z,r,done,status=step(m,s,a,rng,noise)
            history.append((s,a,prob,r))
            if done: break
            if len(history)>1000000: raise RuntimeError('Sonlanmayan bölüm; sahte terminal üretilmedi')
            s=z
        G,logW=0.,0.
        for s,a,b,r in reversed(history):
            G=r+G
            newC=np.logaddexp(C[s][a],logW)
            Q[s][a]+=math.exp(logW-newC)*(G-Q[s][a]); C[s][a]=newC
            policy[s]=greedy(s)
            if a!=policy[s]: break
            logW-=math.log(b)
        returns.append(-len(history))
    return dict(model=m,Q=Q,logC=C,policy=policy,returns=returns)


def trajectory(result, start=0, seed=1, noise=0., max_steps=2000):
    m,rng=result['model'],np.random.default_rng(seed)
    s=m['starts'][start]; path=[s]; events=[]
    for _ in range(max_steps):
        a=result['policy'].get(s,guide(m,s))
        z,r,done,status=step(m,s,a,rng,noise); events.append(status)
        if done: return dict(path=path,events=events,finished=True,steps=len(events))
        path.append(z); s=z
    return dict(path=path,events=events,finished=False,steps=len(events))
