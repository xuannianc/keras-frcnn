# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from frcnn.RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    return '/home/adam/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_feature_map_size(width, height):
    def get_output_size(input_size):
        return input_size // 16
    return get_output_size(width), get_output_size(height)


def base_net(input_tensor=None, trainable=False):
    # Determine proper input shape
    input_shape = (None, None, 3)

    if input_tensor is None:
        image_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            image_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            image_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x


def rpn(base_net_output, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_net_output)
    rpn_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_class')(x)
    rpn_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_regr')(x)
    return [rpn_class, rpn_regr]


def rcnn(base_net_output, rois_input, num_rois, num_classes=21):
    """
    Region-based Convolutional Neural Network
    :param base_net_output: vgg|resnet output
    :param rois_input: 输入的 rois
    :param num_rois: rois 的个数
    :param num_classes:
    :return:
    """
    pool_size = 7
    # shape (batch_size=1,num_rois,pool_size,pool_size,num_channels)
    roi_pool_output = RoiPoolingConv(pool_size, num_rois)([base_net_output, rois_input])
    # TimeDistributed 的功能就是将 out_roi_pool 分成 num_rois 份,分别处理
    out = TimeDistributed(Flatten(name='flatten'))(roi_pool_output)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    rcnn_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer='zero'),
                                name='rcnn_class')(out)
    # note: no regression target for bg class
    rcnn_regr = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='rcnn_regr')(out)
    return [rcnn_class, rcnn_regr]
