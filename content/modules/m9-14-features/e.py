import math
import numpy as np
import torch
import torch.nn.functional as F


def describe(img, kps):
    ...


from skimage import data
kps = torch.tensor([[256.0, 256.0, 2.0, 0.3]], dtype=torch.float64)
print(describe(data.camera(), kps).shape)
