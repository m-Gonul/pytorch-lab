def tconv_out_size(H, k, s, p, op=0):
    ...


def conv_transpose1d(x, w, s=1, p=0):
    ...


def conv_transpose2d(x, w, s=1, p=0):
    ...


def overlap_counts(n, k, s):
    ...


def unet_sizes(H, depth=4):
    ...


def dice_iou(pred, gt):
    ...


print(conv_transpose1d([1, 2, 3], [1, 2, 1], 2))
