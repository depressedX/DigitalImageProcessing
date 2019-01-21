#!/usr/bin/python2.6
# -*- coding: utf-8 -*-

import cv2
import numpy as np

raw_img = cv2.imread('./sample3.jpg')


def equalization(img):
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_gray = converted_img[0:, 0:, 0]
    img_cr = converted_img[0:, 0:, 1]
    img_cb = converted_img[0:, 0:, 2]

    # 划分段数L
    L = 256
    n = np.zeros((L), np.uint16)
    s = np.zeros((L), np.uint16)

    height = converted_img.shape[0]
    width = converted_img.shape[1]

    for i in range(height):
        for j in range(width):
            n[converted_img[i][j][0]] += 1

    sum_nj = 0
    for i in range(L):
        sum_nj += n[i]
        s[i] = round((L - 1) / (height * width) * sum_nj)

    for i in range(height):
        for j in range(width):
            converted_img[i][j][0] = s[converted_img[i][j][0]]

    return cv2.cvtColor(converted_img, cv2.COLOR_YCR_CB2BGR)


converted_img = equalization(raw_img)

cv2.imshow('before', raw_img)
cv2.imshow('after', converted_img)
cv2.waitKey()
