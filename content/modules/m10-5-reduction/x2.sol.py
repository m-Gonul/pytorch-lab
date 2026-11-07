import numpy as np


def warp_serialization(bins):
    b = np.sort(np.asarray(bins), axis=1)
    new = np.concatenate([np.ones((b.shape[0], 1), bool), np.diff(b, axis=1) != 0], 1)
    run_id = np.cumsum(new, axis=1)
    rows = np.broadcast_to(np.arange(b.shape[0])[:, None], b.shape)
    cnt = np.zeros((b.shape[0], 33), np.int64)
    np.add.at(cnt, (rows, run_id), 1)
    return cnt.max(1)


def cost(image, n_bins=256, private=False, block=256):
    v = np.asarray(image).ravel()
    v = v[: len(v) // 32 * 32]
    steps = int(warp_serialization(v.reshape(-1, 32)).sum())
    if private:
        n_blocks = -(-len(v) // block)
        return steps, n_blocks * n_bins
    return steps, 0


print(cost(np.zeros(1024, np.int64)), cost(np.arange(1024) % 256))
