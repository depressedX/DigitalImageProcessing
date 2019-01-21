#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
# 产生噪声图像的频域图像
import cv2
import numpy as np
from matplotlib import pyplot as plt
import fourier_func


# 高斯噪声
def gaussian_noise(img, mu=0, sigma=10):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.normal(mu, sigma, noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    # 叠加
    noise_img += noise
    # 处理溢出的情况
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


# 瑞利噪声
def rayleigh_noise(img, scale=32):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.rayleigh(scale, size=noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    noise_img += noise
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


# 伽马噪声
def gamma_noise(img, shape=2, scale=32):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.gamma(shape, scale, size=noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    noise_img += noise
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


# 指数噪声
def exponential_noise(img, scale=32):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.exponential(scale, size=noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    noise_img += noise
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


def uniform_noise(img, low=0, high=40):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.uniform(low, high, size=noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    noise_img += noise
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


# 脉冲噪声
def pulse_noise(img, low=0, high=40):
    img = np.asarray(img, dtype=np.uint16)
    noise_img = img
    noise = np.random.random_integers(low, high, size=noise_img.shape)
    noise = np.asarray(noise, dtype=np.uint16)
    noise_img += noise
    noise_img = np.minimum(noise_img, 255)
    noise_img = np.maximum(noise_img, 0)
    return noise_img


img = cv2.imread('sample_standard.jpg', 0)
noise_img = uniform_noise(img)

cv2.imwrite('sample_uniform_noise.jpg', noise_img)

# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.title('Input Image')
#
# plt.subplot(122)
# plt.imshow(noise_img, cmap='gray')
# plt.title('frequency domain')

plt.show()
