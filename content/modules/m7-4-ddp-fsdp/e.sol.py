import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 8, generator=g), torch.randn(64, 1, generator=g)


# ---- senin kodun ----

def flatten(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def unflatten(flat, like):
    out, i = [], 0
    for t in like:
        out.append(flat[i:i + t.numel()].view_as(t))
        i += t.numel()
    return out


def zero1_worker(rank, world, port, out, steps, lr=1e-2):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    model = make_model()
    params = list(model.parameters())
    X, y = make_data()
    Xr, yr = X[rank::world], y[rank::world]
    n = sum(p.numel() for p in params)
    chunk = -(-n // world)
    lo, hi = rank * chunk, (rank + 1) * chunk
    m = torch.zeros(chunk)
    v = torch.zeros(chunk)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        for p in params:
            p.grad = None
        ((model(Xr) - yr) ** 2).mean().backward()
        g = torch.zeros(chunk * world)
        g[:n] = flatten([p.grad for p in params])
        dist.all_reduce(g)
        g /= world
        w = torch.zeros(chunk * world)
        w[:n] = flatten([p.detach() for p in params])
        gs, ws = g[lo:hi], w[lo:hi]
        m.mul_(b1).add_((1 - b1) * gs)
        v.mul_(b2).add_((1 - b2) * gs * gs)
        ws = ws - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        parts = [torch.empty(chunk) for _ in range(world)]
        dist.all_gather(parts, ws.contiguous())
        new = torch.cat(parts)[:n]
        with torch.no_grad():
            for p, q in zip(params, unflatten(new, params)):
                p.copy_(q)
    torch.save(([p.detach().clone() for p in params], m.numel() + v.numel()), os.path.join(out, f"r{rank}.pt"))
    dist.destroy_process_group()


def run_zero1(world, steps=10):
    out = tempfile.mkdtemp()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.start_processes(zero1_worker, args=(world, port, out, steps), nprocs=world, start_method="fork")
    res = [torch.load(os.path.join(out, f"r{r}.pt")) for r in range(world)]
    return [r[0] for r in res], [r[1] for r in res]


P, sizes = run_zero1(2)
print(sizes)
