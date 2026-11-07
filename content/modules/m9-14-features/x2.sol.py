import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC


def shapes(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 64, 64), np.uint8)
    y = rng.integers(0, 2, n)
    for i in range(n):
        bg, fg = rng.integers(0, 256, 2)
        while abs(int(bg) - int(fg)) < 60:
            fg = rng.integers(0, 256)
        img = np.full((64, 64), int(bg), np.uint8)
        r = int(rng.integers(8, 18)); cx, cy = rng.integers(r + 2, 62 - r, 2)
        if y[i] == 0:
            cv2.circle(img, (int(cx), int(cy)), r, int(fg), -1)
        else:
            cv2.rectangle(img, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), int(fg), -1)
        X[i] = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    return X, y


def hog_features(X):
    return np.stack([hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys") for x in X])


def train_eval(Xtr, ytr, Xte, yte):
    h = LinearSVC(C=0.01).fit(hog_features(Xtr), ytr).score(hog_features(Xte), yte)
    r = LinearSVC(C=0.01).fit(Xtr.reshape(len(Xtr), -1) / 255, ytr).score(Xte.reshape(len(Xte), -1) / 255, yte)
    return h, r


X, y = shapes(10, 0)
print(X.shape, hog_features(X).shape)
