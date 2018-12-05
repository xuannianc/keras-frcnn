from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

# loss 的系数
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_regr_loss(num_anchors):
    def rpn_regr_loss_fixed_num(y_true, y_pred):
        # :4 * num_anchors 部分是 is_box_valid, valid 为 1, invalid 为 0
        # y_true[:,:,:,:4*num_anchors] 作为乘式因子就是想忽略 invalid 这部分的 loss
        diff = y_true[:, :, :, 4 * num_anchors:] - y_pred
        abs_diff = K.abs(diff)
        # less_equal 逐元素比较 a<=b, less 逐元素比较 a<b
        bool_diff = K.cast(K.less_equal(abs_diff, 1.0), tf.float32)
        # smooth_L1_loss = 0.5 * x * x if |x| < 1 else |x| - 0.5
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (
                    bool_diff * (0.5 * diff * diff) + (1 - bool_diff) * (abs_diff - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_regr_loss_fixed_num


def rpn_class_loss(num_anchors):
    def rpn_class_loss_fixed_num(y_true, y_pred):
        # y_true[:, :, :, :num_anchors] 表示的是 anchor 是否 is valid, 既是否是 neutral
        # y_true[:, :, :, num_anchors:] 表示的是 anchor 是否 overlap, 既是 positive,还是 negative
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                                              y_true[:, :, :,
                                                                                              num_anchors:])) / K.sum(
            epsilon + y_true[:, :, :, :num_anchors])

    return rpn_class_loss_fixed_num


def rcnn_regr_loss(num_classes):
    def rcnn_regr_loss_fixed_num(y_true, y_pred):
        # y_true 的 [4*num_classes:] 部分是预测的回归值,[:4*num_classes] 部分是标记对应的位置是否要参与 loss 计算
        diff = y_true[:, :, 4 * num_classes:] - y_pred
        abs_diff = K.abs(diff)
        bool_diff = K.cast(K.less_equal(abs_diff, 1.0), 'float32')
        return lambda_cls_regr * K.sum(
            y_true[:, :, :4 * num_classes] * (
                    bool_diff * (0.5 * diff * diff) + (1 - bool_diff) * (abs_diff - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])

    return rcnn_regr_loss_fixed_num


def rcnn_class_loss(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
