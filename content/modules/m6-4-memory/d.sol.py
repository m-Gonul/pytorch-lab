import torch
from torch.utils.checkpoint import checkpoint


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def optimizer_state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


def saved_bytes(fn):
    tot = [0]

    def pack(t):
        tot[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, tot[0]


def run_checkpointed(blocks, x, k):
    def seg(h, s):
        for b in blocks[s:s + k]:
            h = b(h)
        return h
    h = x
    for s in range(0, len(blocks), k):
        h = checkpoint(seg, h, s, use_reentrant=False)
    return h


torch.manual_seed(0)
blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh()) for _ in range(16))
x = torch.randn(64, 256, requires_grad=True)
print(param_bytes(blocks))
print(saved_bytes(lambda: run_checkpointed(blocks, x, 4))[1])   # 4 segment girdisi × 65536
