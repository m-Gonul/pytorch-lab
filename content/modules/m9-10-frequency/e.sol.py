import torch
import torch.nn.functional as F


def conv2d_fft(x, w, padding=0):
    if padding:
        x = F.pad(x, (padding,) * 4)
    kh, kw = w.shape[-2:]
    Hp, Wp = x.shape[-2:]
    s = (Hp + kh - 1, Wp + kw - 1)
    X = torch.fft.rfft2(x, s=s)
    Wf = torch.fft.rfft2(w.flip(-2, -1), s=s)
    y = torch.fft.irfft2(torch.einsum("bihw,oihw->bohw", X, Wf), s=s)
    return y[..., kh - 1:Hp, kw - 1:Wp]


x = torch.randn(1, 2, 16, 16); w = torch.randn(3, 2, 5, 5)
print((conv2d_fft(x, w, 2) - F.conv2d(x, w, padding=2)).abs().max())
