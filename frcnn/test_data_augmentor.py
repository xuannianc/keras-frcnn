import cv2
import numpy as np


def test_flip(image):
    # cv2.flip 第二个参数, 0: 竖直翻转 1:水平翻转 -1:竖直水平翻转
    flipped_image = cv2.flip(image, -1)
    cv2.imshow('image', image)
    cv2.imshow('flipped_image', flipped_image)
    cv2.waitKey(0)


# clockwise rotation
def test_rotate(image):
    transposed_image = np.transpose(image, (1, 0, 2))
    # 相当于 先顺时针旋转 270°,再竖直翻转
    # 或者 先顺时针旋转 90°,再水平翻转
    cv2.imshow('image', image)
    cv2.imshow('transposed_image', transposed_image)
    cv2.waitKey(0)
    # 顺时针旋转 270°
    rotated_image_270 = cv2.flip(transposed_image, 0)
    print('shape_270={}'.format(rotated_image_270.shape))
    # 顺时针旋转 180°
    rotated_image_180 = cv2.flip(image, -1)
    print('shape_180={}'.format(rotated_image_180.shape))
    # 顺时针旋转 90°
    rotated_image_90 = cv2.flip(transposed_image, 1)
    print('shape_90={}'.format(rotated_image_90.shape))
    cv2.imshow('image', image)
    cv2.imshow('rotated_image_270', rotated_image_270)
    cv2.imshow('rotated_image_180', rotated_image_180)
    cv2.imshow('rotated_image_90', rotated_image_90)
    cv2.waitKey(0)


image = cv2.imread('test.jpg')
# test_flip(image)
test_rotate(image)
