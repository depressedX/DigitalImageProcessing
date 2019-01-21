#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
# 空域到频域
import cv2
import numpy as np
from matplotlib import pyplot as plt
import fourier_func

img = cv2.imread('rect.jpg', 0)

f1 = fourier_func.fft2(img)
f2 = np.fft.fft2(img)
# f1 = f2

print(np.allclose(f1, f2))

f1shift = np.fft.fftshift(f1.real)
f2shift = np.fft.fftshift(f2.real)
f1final = 20 * np.log(np.abs(f1))
f2final = 20 * np.log(np.abs(f2))

plt.subplot(211)
plt.imshow(img, cmap='gray')
plt.title('Input Image 256 X 128')

plt.subplot(223)
plt.imshow(f1final, cmap='gray')
plt.title('my alogorithm')

plt.subplot(224)
plt.imshow(f2final, cmap='gray')
plt.title('lib alogithm')

plt.show()
