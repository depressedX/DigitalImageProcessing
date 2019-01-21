
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import time

def dft2(img):
    img = np.asarray(img, dtype=np.float)
    height, width = img.shape

    fft_img = np.zeros((height, width), dtype=np.complex)

    for w in range(width):
        for h in range(height):

            for rw in range(width):
                for rh in range(height):
                    t = img[rh][rw] * (np.exp(-2j * math.pi * (h * rh / height + w * rw / width)))
                    fft_img[h][w] += t
    return fft_img

def fft2(img):
    img = np.asarray(img, dtype=np.float)
    N = img.shape[0]
    M = img.shape[1]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:
        # 枚举另一个维度

        if M % 2 > 0:
            raise ValueError("size of y must be a power of 2")
        elif M <= 8:
            # 应用dft2
            return dft2(img)
        else:
            # 二分
            Y_even = fft2(img[:, ::2])
            Y_odd = fft2(img[:, 1::2])
            factor = np.zeros((N, M),dtype=complex)
            for i in range(factor.shape[0]):
                # for j in range(factor.shape[1]):
                factor[i, :] = np.exp(-2j * np.pi * np.arange(M) / M)
            return np.concatenate([Y_even + factor[:, :M // 2] * Y_odd,
                                   Y_even + factor[:, M // 2:] * Y_odd], axis=1)
    else:
        # 二分
        X_even = fft2(img[::2, :])
        X_odd = fft2(img[1::2, :])
        factor = np.zeros((N, M),dtype=complex)
        for i in range(factor.shape[1]):
            factor[:, i] = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd], axis=0)

img = cv2.imread('rect.jpg', 0)
s1 = time.clock()
f1 = fft2(img)
print(time.clock()-s1)
s1 = time.clock()
f2 = np.fft.fft2(img)
print(time.clock()-s1)
# f1 = f2
print(np.allclose(f1,f2))
f1shift = np.fft.fftshift(f1.real)
f2shift = np.fft.fftshift(f2.real)
f1final = 20 * np.log(np.abs(f1shift))
f2final = 20 * np.log(np.abs(f2shift))

print(f1final[0:4,0:4])
print(f2final[0:4,0:4])
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
