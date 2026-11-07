import torch


def equalize_batch(imgs):
    """imgs: (B, H, W) uint8. OpenCV varyantı. Döngü yok."""
    ...


imgs = torch.randint(30, 120, (4, 64, 64), dtype=torch.uint8)
print(equalize_batch(imgs).float().mean((1, 2)))
