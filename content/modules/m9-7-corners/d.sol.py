import math
import torch
import torch.nn.functional as F


def similarity(img, angle, scale):
    """Merkez etrafında döndür + ölçekle. (img2, A) döndürür; A (2, 3): p2 = A @ [x, y, 1]."""
    H, W = img.shape
    c = torch.tensor([W / 2, H / 2])
    th = math.radians(angle)
    Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * scale
    inv = torch.linalg.inv(Rm)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    src = (torch.stack([xs, ys], -1) - c) @ inv.T + c
    grid = torch.stack([src[..., 0] / (W - 1) * 2 - 1, src[..., 1] / (H - 1) * 2 - 1], -1)[None]
    img2 = F.grid_sample(img[None, None], grid, align_corners=True)[0, 0]
    return img2, torch.cat([Rm, (c - Rm @ c)[:, None]], 1)


def _gauss(s):
    r = int(math.ceil(3 * s))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-t ** 2 / (2 * s * s))
    return g / g.sum()


def _sep(x, g):
    r = len(g) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))


def harris_torch(img, sigma_d=1.0, sigma_i=2.0, k=0.04):
    x = _sep(img[None, None].float(), _gauss(sigma_d))
    d = torch.tensor([-0.5, 0.0, 0.5])
    gx = F.conv2d(F.pad(x, (1, 1, 0, 0), mode="replicate"), d.view(1, 1, 1, 3))
    gy = F.conv2d(F.pad(x, (0, 0, 1, 1), mode="replicate"), d.view(1, 1, 3, 1))
    g = _gauss(sigma_i)
    a, b, c = _sep(gx * gx, g), _sep(gx * gy, g), _sep(gy * gy, g)
    return (a * c - b * b - k * (a + c) ** 2)[0, 0]


def detect(R, n=200, r=4, border=16):
    mx = F.max_pool2d(R[None, None], 2 * r + 1, 1, r)[0, 0]
    m = R == mx
    m[:border] = False
    m[-border:] = False
    m[:, :border] = False
    m[:, -border:] = False
    s = torch.where(m, R, torch.tensor(-float("inf")))
    _, i = s.flatten().topk(n)
    W = R.shape[1]
    return torch.stack([i % W, i // W], 1).float()


def repeatability(img, angle, scale, adapt=False, tol=2.0):
    H, W = img.shape
    img2, A = similarity(img, angle, scale)
    p1 = detect(harris_torch(img))
    if adapt:
        p2 = detect(harris_torch(img2, scale, 2 * scale), r=int(4 * scale))
    else:
        p2 = detect(harris_torch(img2))
    q = p1 @ A[:, :2].T + A[:, 2]
    inside = (q[:, 0] > 16) & (q[:, 0] < W - 16) & (q[:, 1] > 16) & (q[:, 1] < H - 16)
    d = torch.cdist(q[inside], p2).min(1).values
    return (d <= tol).float().mean().item()


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32) / 255
print(repeatability(img, 30, 1.0))
