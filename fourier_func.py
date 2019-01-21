import numpy as np
import math


def dft2(img):
    # 将int16转为float
    img = np.asarray(img, dtype=np.float)
    height, width = img.shape

    # 转换为频域空间后变为复数
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
            # 二分 奇数偶数部分
            Y_even = fft2(img[:, ::2])
            Y_odd = fft2(img[:, 1::2])
            factor = np.zeros((N, M), dtype=complex)
            for i in range(factor.shape[0]):
                factor[i, :] = np.exp(-2j * np.pi * np.arange(M) / M)
            # 在M/2之后的值X(n)根据周期性和X(n-M/2)相同
            return np.concatenate([Y_even + factor[:, :M // 2] * Y_odd,
                                   Y_even + factor[:, M // 2:] * Y_odd], axis=1)
    else:
        # 二分
        X_even = fft2(img[::2, :])
        X_odd = fft2(img[1::2, :])
        factor = np.zeros((N, M), dtype=complex)
        for i in range(factor.shape[1]):
            factor[:, i] = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd], axis=0)


# 一维离散傅里叶变换
def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


# 一维快速傅里叶变换
def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])


# 一维离散傅里叶逆变换
def idft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, x) / N


# 一维快速傅里叶逆变换
def ifft(x):
    x = np.asarray(x, dtype=np.complex)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 2:
        return idft(x)
    else:
        X_even = ifft(x[::2] * np.exp(2j * np.pi * np.arange(N // 2)))
        X_odd = ifft(x[1::2] * np.exp(2j * np.pi * np.arange(N // 2)))
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd]) / 2


def idft2(f_img):
    height, width = f_img.shape

    raw_img = np.zeros((height, width), dtype=np.complex)

    for w in range(width):
        for h in range(height):

            for rw in range(width):
                for rh in range(height):
                    t = f_img[rh][rw] * (np.exp(2j * math.pi * (h * rh / height + w * rw / width)))
                    raw_img[h][w] += t
    raw_img /= height * width
    return raw_img


def ifft2(f_img):
    N = f_img.shape[0]
    M = f_img.shape[1]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:
        # 枚举另一个维度

        if M % 2 > 0:
            raise ValueError("size of y must be a power of 2")
        elif M <= 8:
            # 应用dft2
            return idft2(f_img)
        else:
            # 二分
            x_factor = np.zeros((N, M // 2), dtype=np.complex)
            for i in range(N):
                x_factor[i] = np.exp(2j * np.pi * np.arange(M // 2))
            Y_even = ifft2(f_img[:, ::2] * x_factor)
            Y_odd = ifft2(f_img[:, 1::2] * x_factor)
            factor = np.zeros((N, M), dtype=complex)
            for i in range(factor.shape[0]):
                factor[i, :] = np.exp(2j * np.pi * np.arange(M) / M)
            return np.concatenate([Y_even + factor[:, :M // 2] * Y_odd,
                                   Y_even + factor[:, M // 2:] * Y_odd], axis=1) / 2
    else:
        # 二分
        x_factor = np.zeros((N // 2, M), dtype=np.complex)
        for i in range(M):
            x_factor[:, i] = np.exp(2j * np.pi * np.arange(N // 2))
        X_even = ifft2(f_img[::2, :] * x_factor)
        X_odd = ifft2(f_img[1::2, :] * x_factor)
        factor = np.zeros((N, M), dtype=complex)
        for i in range(factor.shape[1]):
            factor[:, i] = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd], axis=0) / 2
