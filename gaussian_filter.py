#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from functools import reduce


def gaussian(img, sigma=1.0):
    n, m = img.shape
    print(n, m)
    g = lambda x: math.exp(-x * x / (2 * sigma * sigma))

    X = np.asarray(img)
    Z = np.array(X, dtype=np.float)
    Y = np.zeros((n, m), dtype=np.float)
    # 窗口尺寸
    k_size = math.floor(6 * sigma - 1) // 2 * 2 + 1
    w = k_size // 2
    Sp = reduce(lambda x, y: x + y, np.exp(-np.square(np.arange(-w, w + 1)) / (2 * np.square(sigma))))

    # 横向扩展X 镜像
    X = np.concatenate([np.flip(X[:, 1:w + 1], axis=1), X, np.flip(X[:, -w - 1:m - 1], axis=1)], axis=1)
    # 横向计算
    for i in range(n):
        for j in range(m):
            Z[i, j] = 0
            for v in range(-w, w + 1):
                Z[i, j] += X[i, j + v] * g(v) / Sp

    print(Z.shape)
    # 纵向计算
    # 纵向扩展Z
    Z = np.concatenate([np.flip(Z[1:w + 1, :], axis=0), Z, np.flip(Z[-w - 1:n - 1, :], axis=0)], axis=0)
    print(Z.shape)
    for i in range(n):
        for j in range(m):
            Y[i, j] = 0
            for u in range(-w, w + 1):
                Y[i, j] += Z[i + u, j] * g(u) / Sp

    return Z


img = cv2.imread('test.jpg', 0)

g1_img = gaussian(img,sigma=1)
# g2_img = gaussian(img,sigma=1)
# g3_img = gaussian(img,sigma=2)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

plt.subplot(222)
plt.imshow(g1_img, cmap='gray')
plt.title('sigma = 0.5')
#
# plt.subplot(223)
# plt.imshow(g2_img, cmap='gray')
# plt.title('sigma = 1')
#
# plt.subplot(224)
# plt.imshow(g3_img, cmap='gray')
# plt.title('sigma = 2')

plt.show()
