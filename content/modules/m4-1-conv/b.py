def pad2d(x, p):
    ...


def conv2d_loops(x, w, b, stride=1, pad=0):
    ...


x = [[[1, 2, 0], [0, 1, 3], [2, 1, 0]]]
w = [[[[1, 0], [-1, 2]]]]
print(conv2d_loops(x, w, [0]))      # [[[3, 7], [0, 0]]]
