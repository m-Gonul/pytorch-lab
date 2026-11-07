import torch


def bf16_saved(fn, *args):
    """fn(*args); saklanan kayan noktalı tensörler bf16'da tutulur."""
    def pack(t):
        if t.is_floating_point():
            return (t.dtype, t.to(torch.bfloat16))
        return (None, t)

    def unpack(p):
        dtype, t = p
        return t if dtype is None else t.to(dtype)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return fn(*args)


torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256))
x = torch.randn(128, 256)
bf16_saved(mlp, x).pow(2).mean().backward()
print(mlp[0].weight.grad.norm())
