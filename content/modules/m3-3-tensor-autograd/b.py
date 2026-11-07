def broadcast_shape(s1, s2):
    ...


def unbroadcast2d(G, shape):
    ...


print(broadcast_shape((2, 3), (3,)), broadcast_shape((4, 1), (1, 5)))   # (2, 3) (4, 5)
G = [[1, 2, 3], [4, 5, 6]]
print(unbroadcast2d(G, (3,)), unbroadcast2d(G, (2, 1)), unbroadcast2d(G, ()))   # [5, 7, 9] [[6], [15]] 21
