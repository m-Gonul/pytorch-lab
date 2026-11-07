import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def estimate_motion(frames):
    out = []
    for a, b in zip(frames[:-1], frames[1:]):
        a8, b8 = np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)
        p = cv2.goodFeaturesToTrack(a8, 300, 0.01, 10)
        q, st, _ = cv2.calcOpticalFlowPyrLK(a8, b8, p, None)
        ok = st[:, 0] == 1
        M, _ = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC)
        out.append([M[0, 2], M[1, 2], math.atan2(M[1, 0], M[0, 0])])
    return np.array(out)


def smooth(traj, r=5):
    P = np.pad(traj, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(np.pad(P, ((1, 0), (0, 0))), 0)
    return (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)


def stabilize(frames, r=5):
    frames = np.asarray(frames, np.float32)
    T, H, W = frames.shape
    traj = np.vstack([[0, 0, 0], np.cumsum(estimate_motion(frames), 0)])
    corr = smooth(traj, r) - traj
    dx, dy, da = (torch.tensor(corr[:, i]) for i in range(3))
    M = torch.zeros(T, 3, 3, dtype=torch.float64)
    M[:, 0, 0] = torch.cos(da); M[:, 0, 1] = -torch.sin(da); M[:, 0, 2] = dx
    M[:, 1, 0] = torch.sin(da); M[:, 1, 1] = torch.cos(da); M[:, 1, 2] = dy
    M[:, 2, 2] = 1
    N = torch.tensor([[2 / (W - 1), 0, -1], [0, 2 / (H - 1), -1], [0, 0, 1]], dtype=torch.float64)
    theta = (N @ torch.linalg.inv(M) @ torch.linalg.inv(N))[:, :2]
    grid = F.affine_grid(theta.float(), (T, 1, H, W), align_corners=True)
    out = F.grid_sample(torch.tensor(frames)[:, None], grid, align_corners=True)[:, 0]
    return out.numpy(), corr


print("stabilizasyon")
