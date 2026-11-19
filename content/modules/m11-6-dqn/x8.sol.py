import torch


def residual_step(w, x, xp1, reward1, xp2=None, gamma=1.0, alpha=.1):
    param = w.detach().clone().requires_grad_(True)
    delta = reward1 + gamma*(xp1@param) - x@param
    loss = .5*delta.square() if xp2 is None else -delta.detach()*((x-gamma*xp2)@param)
    gradient, = torch.autograd.grad(loss, param)
    return (param-alpha*gradient).detach()


def expected_split(w, presplit=False, method='naive'):
    X = torch.eye(3, dtype=w.dtype, device=w.device)
    if presplit:
        states = [(X[0], [(X[1],0.,1.)], .25), (X[0], [(X[2],0.,1.)], .25)]
    else:
        states = [(X[0], [(X[1],0.,.5), (X[2],0.,.5)], .5)]
    zero = torch.zeros_like(w)
    states += [(X[1],[(zero,1.,1.)],.25), (X[2],[(zero,0.,1.)],.25)]
    update = torch.zeros_like(w)
    for x, transitions, weight in states:
        for xp, reward, prob in transitions:
            delta = reward + xp@w-x@w
            if method == 'naive':
                update += weight*prob*delta*(x-xp)
            elif method == 'double':
                for xp2, _, prob2 in transitions:
                    update += weight*prob*prob2*delta*(x-xp2)
            else:
                raise ValueError("method naive veya double olmalı")
    return update


w = torch.tensor([.5,.75,.25],dtype=torch.float64)
print(expected_split(w,False,'naive'))
print(expected_split(w,False,'double'))
