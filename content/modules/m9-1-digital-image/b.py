def bilinear_at(img, x, y):
    ...


def resize_nearest(img, H2, W2):
    ...


def resize_bilinear(img, H2, W2):
    ...


I = [[10.0, 20.0], [30.0, 40.0]]
print(bilinear_at(I, 0.25, 0.5))          # 22.5
print(resize_bilinear(I, 4, 4)[0])
