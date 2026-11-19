import torch


def residual_step(w, x, xp1, reward1, xp2=None, gamma=1.0, alpha=.1):
    # TODO: tek örnekli tam türev veya bağımsız ikinci örnekli BE yönü.
    raise NotImplementedError


def expected_split(w, presplit=False, method='naive'):
    # TODO: µ ağırlıklarıyla, aynı gerçek duruma koşullu bütün olasılıkları sayın.
    raise NotImplementedError


w = torch.tensor([.5,.75,.25],dtype=torch.float64)
print(expected_split(w,False,'naive'))
print(expected_split(w,False,'double'))
