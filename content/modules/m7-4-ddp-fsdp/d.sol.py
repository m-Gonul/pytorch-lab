import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 8, generator=g), torch.randn(64, 1, generator=g)


# ---- senin kodun ----

def allreduce_mean(t):
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def worker(rank, world, port, out, steps, manual):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = make_model()
    net = model if manual else DDP(model)
    X, y = make_data()
    Xr, yr = X[rank::world], y[rank::world]
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(steps):
        loss = ((net(Xr) - yr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if manual:
            for p in model.parameters():
                allreduce_mean(p.grad)
        opt.step()
    torch.save([p.detach().clone() for p in model.parameters()], os.path.join(out, f"r{rank}.pt"))
    dist.destroy_process_group()


def run(world, steps=5, manual=False):
    out = tempfile.mkdtemp()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.start_processes(worker, args=(world, port, out, steps, manual), nprocs=world, start_method="fork")
    return [torch.load(os.path.join(out, f"r{r}.pt")) for r in range(world)]


P = run(2)
print(all(torch.equal(a, b) for a, b in zip(P[0], P[1])))
