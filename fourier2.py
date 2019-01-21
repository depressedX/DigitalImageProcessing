#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
# 频域到空域
import cv2
import numpy as np
from matplotlib import pyplot as plt
import fourier_func

img = cv2.imread('sample_s_standard.jpg', 0)

f = fourier_func.fft2(img)
f_img = 20 * np.log(np.abs(np.fft.fftshift(f).real))
i = fourier_func.ifft2(f)
i_img = np.abs(i)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

plt.subplot(132)
plt.imshow(f_img, cmap='gray')
plt.title('frequency domain')

plt.subplot(133)
plt.imshow(i_img, cmap='gray')
plt.title('frequency domain')


plt.show()
