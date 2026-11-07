import torch
import torch.nn.functional as F


def accum_step(model, opt, x, y, micro):
    N = x.shape[0]
    for xs, ys in zip(x.split(micro), y.split(micro)):
        loss = F.cross_entropy(model(xs), ys) * xs.shape[0] / N
        loss.backward()
    opt.step()
    opt.zero_grad()


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(70, 20), torch.randint(0, 5, (70,))
accum_step(model, opt, x, y, micro=16)
print(F.cross_entropy(model(x), y).item())
