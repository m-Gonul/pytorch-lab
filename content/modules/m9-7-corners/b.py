def harris_matrix(patch):
    ...


def harris_response(img, k, r=1):
    ...


def local_maxima(R, thr, r):
    ...


left = [[0, 255, 255, 255, 255], [0, 0, 255, 255, 255], [0, 0, 0, 255, 255], [0, 0, 0, 0, 255], [0, 0, 0, 0, 0]]
print(harris_matrix(left))
