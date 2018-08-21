import cv2
import numpy as np


def test_flip(image):
    flipped_image = cv2.flip(image, -1)
    cv2.imshow('image', image)
    cv2.imshow('flipped_image', flipped_image)
    cv2.waitKey(0)


# clockwise rotation
def test_rotate(image):
    rotated_image_270 = np.transpose(image, (1, 0, 2))
    cv2.imshow('rotated_image_270_1', rotated_image_270)
    cv2.waitKey(0)
    rotated_image_270 = cv2.flip(rotated_image_270, 0)
    print('shape_270={}'.format(rotated_image_270.shape))
    rotated_image_180 = cv2.flip(image, -1)
    print('shape_180={}'.format(rotated_image_180.shape))
    rotated_image_90 = np.transpose(image, (1, 0, 2))
    rotated_image_90 = cv2.flip(rotated_image_90, 1)
    print('shape_90={}'.format(rotated_image_90.shape))
    cv2.imshow('image', image)
    cv2.imshow('rotated_image_270_2', rotated_image_270)
    # cv2.imshow('rotated_image_180', rotated_image_180)
    # cv2.imshow('rotated_image_90', rotated_image_90)
    cv2.waitKey(0)


image = cv2.imread('pikachu.jpg')
# test_flip(image)
test_rotate(image)
