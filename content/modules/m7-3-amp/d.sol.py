import torch
import torch.nn.functional as F


def amp_dtypes(model, x, y, device_type, dtype):
    with torch.autocast(device_type, dtype=dtype):
        out = model(x)
        loss = F.cross_entropy(out, y)
    loss.backward()
    p = next(model.parameters())
    res = {"linear": out.dtype, "loss": loss.dtype, "param": p.dtype, "grad": p.grad.dtype}
    model.zero_grad()
    return res


def train_step_amp(model, opt, scaler, x, y, device_type, dtype):
    with torch.autocast(device_type, dtype=dtype):
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    return loss.item()


dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.bfloat16
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)).to(dev)
x, y = torch.randn(32, 20, device=dev), torch.randint(0, 3, (32,), device=dev)
print(amp_dtypes(model, x, y, dev, dt))
