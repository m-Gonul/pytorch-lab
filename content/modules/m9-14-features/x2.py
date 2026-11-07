import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC


def shapes(n, seed):
    ...


def hog_features(X):
    ...


def train_eval(Xtr, ytr, Xte, yte):
    ...


X, y = shapes(10, 0)
print(X.shape, hog_features(X).shape)
