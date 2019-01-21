import cv2
import numpy as np
import math

raw_img = cv2.imread('./sample.jpg')

initial_shape = raw_img.shape


# 顺时针旋转图像 deg为弧度
def rotate(img, deg):
    # 原图的宽高
    height = img.shape[0]
    width = img.shape[1]
    # 转换后的宽高
    a_width = math.floor(width * math.cos(deg) + height * math.sin(deg))
    a_height = math.floor(height * math.cos(deg) + width * math.sin(deg))
    # 转换后的图像
    converted_img = np.zeros((a_height, a_width, 3), np.uint8)
    # 遍历像素
    for row in range(a_height):
        for col in range(a_width):
            # 坐标转换后的 旋转后的像素坐标
            x1 = col - a_width / 2
            y1 = a_height / 2 - row
            # 坐标转换后的 旋转前的像素坐标
            x0 = x1 * math.cos(deg) - y1 * math.sin(deg)
            y0 = x1 * math.sin(deg) + y1 * math.cos(deg)
            # 原始像素坐标
            x0 = x0 + width / 2
            y0 = height / 2 - y0
            if 0 <= x0 <= width - 1 and 0 <= y0 <= height - 1:
                # 因为得到的像素坐标是浮点数 需要经过线性插值计算
                u = x0 - math.floor(x0)
                v = y0 = math.floor(y0)
                r1 = img[math.floor(y0)][math.floor(x0)] * (1 - u) \
                     + img[math.floor(y0)][math.ceil(x0)] * u
                r2 = img[math.ceil(y0)][math.floor(x0)] * (1 - u) \
                     + img[math.ceil(y0)][math.ceil(x0)] * u
                converted_img[row][col] = r1 * (1 - v) + r2 * v
    return converted_img


# 缩放  <1 缩小  >1 放大
def scale(img, scaleX, scaleY):
    # 原图的宽高
    height = img.shape[0]
    width = img.shape[1]
    # 转换后的宽高
    a_width = math.floor(width * scaleX)
    a_height = math.floor(height * scaleY)
    # 转换后的图像
    converted_img = np.zeros((a_height, a_width, 3), np.uint8)
    for row in range(a_height):
        for col in range(a_width):
            x0 = col / scaleX
            y0 = row / scaleY
            if 0 <= x0 <= width - 1 and 0 <= y0 <= height - 1:
                # 因为得到的像素坐标是浮点数 需要经过线性插值计算
                u = x0 - math.floor(x0)
                v = y0 = math.floor(y0)
                r1 = img[math.floor(y0)][math.floor(x0)] * (1 - u) \
                     + img[math.floor(y0)][math.ceil(x0)] * u
                r2 = img[math.ceil(y0)][math.floor(x0)] * (1 - u) \
                     + img[math.ceil(y0)][math.ceil(x0)] * u
                converted_img[row][col] = r1 * (1 - v) + r2 * v
    return converted_img


scaled_img = scale(raw_img, .8, .8)
cv2.imshow('scale', scaled_img)
cv2.imwrite('scaled_sample.jpg',scaled_img)
rotated_img = rotate(raw_img, math.pi/6)
cv2.imshow('rotate', rotated_img)
cv2.imwrite('rotated_sample.jpg',rotated_img)

cv2.imshow('raw', raw_img)
cv2.waitKey()
