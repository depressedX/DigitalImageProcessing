#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
# 基于积分图的快速均值滤波
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def mean_filter(img, size):
    img = np.asarray(img)
    n, m = img.shape
    # 对原图边界进行扩展  向前扩展size+1  向后扩展size  对称填充
    # 纵向
    img = np.concatenate([np.flip(img[1:size + 2, :], axis=0), img, np.flip(img[-size - 1:n - 1, :], axis=0)], axis=0)
    # 横向
    img = np.concatenate([np.flip(img[:, 1:size + 2], axis=1), img, np.flip(img[:, -size - 1:m - 1], axis=1)], axis=1)

    S = np.zeros(img.shape)
    sum = np.zeros(img.shape)
    S[0, 0] = sum[0, 0] = img[0, 0]
    for row in range(1, img.shape[0]):
        S[row, 0] = sum[row, 0] = sum[row - 1][0] + img[row, 0]
    for col in range(1, img.shape[1]):
        S[0, col] = sum[0, col] = S[0, col - 1] + img[0, col]
        sum[0, col] = img[0, col]
    for row in range(1, img.shape[0]):
        for col in range(1, img.shape[1]):
            sum[row, col] = sum[row - 1, col] + img[row, col]
            S[row, col] = S[row, col - 1] + sum[row, col]
    filter_img = np.zeros((n,m), dtype=np.uint16)

    for row in range(filter_img.shape[0]):
        for col in range(filter_img.shape[1]):
            s_row = row + size + 1
            s_col = col + size + 1
            filter_img[row, col] = (S[s_row + size, s_col + size] + S[s_row - size - 1, s_col - size - 1] - S[
                s_row + size, s_col - size - 1] - S[s_row - size - 1, s_col + size]) / math.pow(2 * size + 1, 2)
    return filter_img


img = cv2.imread('sample.jpg', 0)

g1_img = mean_filter(img, size=1)
g2_img = mean_filter(img,size=2)
g3_img = mean_filter(img,size=3)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

plt.subplot(222)
plt.imshow(g1_img, cmap='gray')
plt.title('size = 1')
#
plt.subplot(223)
plt.imshow(g2_img, cmap='gray')
plt.title('sigma = 2')
#
plt.subplot(224)
plt.imshow(g3_img, cmap='gray')
plt.title('sigma = 3')

plt.show()
