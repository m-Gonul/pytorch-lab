def histogram(img):
    ...


def equalize_class(img):
    ...


def equalize_opencv(img):
    ...


img = [[0, 0, 1, 1], [2, 2, 2, 3]]
print(histogram(img)[:4], equalize_class(img), equalize_opencv(img))
