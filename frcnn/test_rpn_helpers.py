import json
from log import logger
import cv2
import frcnn.resnet as nn
from keras.layers import Input
from keras.models import Model
import numpy as np
from frcnn.roi_helpers_for_test import rpn_to_roi, calc_iou, apply_regr
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test_rpn_to_rois():
    pass


def format_image_size(image, C):
    """ formats the image size based on config """
    image_min_size = C.image_min_size
    (height, width, _) = image.shape

    if width <= height:
        ratio = image_min_size / width
        new_height = int(ratio * height)
        new_width = int(image_min_size)
    else:
        ratio = image_min_size / height
        new_width = int(ratio * width)
        new_height = int(image_min_size)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image, ratio


def format_image_channels(image, C):
    """ formats the image channels based on config """
    image = image.astype(np.float32)
    image[:, :, 0] -= C.image_channel_mean[0]
    image[:, :, 1] -= C.image_channel_mean[1]
    image[:, :, 2] -= C.image_channel_mean[2]
    image /= C.image_scaling_factor
    image_input = np.expand_dims(image, axis=0)
    return image_input


def format_image(image, C):
    """ formats an image for model prediction based on config """
    image, ratio = format_image_size(image, C)
    src_image = image.copy()
    image_input = format_image_channels(image, C)
    return image_input, ratio, src_image


def show_rois_with_probs(image, rois, probs, num_show_rois=5):
    roi_idxs = np.argsort(probs)[::-1]
    colors = [np.random.randint(0, 255, 3) for i in range(num_show_rois)]
    for start_idx in range(0, 300, num_show_rois):
        image_rois_with_probs = image.copy()
        for idx, roi_idx in enumerate(roi_idxs[start_idx:start_idx + num_show_rois]):
            x1, y1, x2, y2 = rois[roi_idx] * C.rpn_stride
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            cv2.rectangle(image_rois_with_probs, (x1, y1), (x2, y2), colors[idx].tolist(), 2)
            cv2.putText(image_rois_with_probs, '{:.2f}'.format(probs[roi_idx]), (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 0, 255), 1)
        cv2.namedWindow('image_rois_with_probs', cv2.WINDOW_NORMAL)
        cv2.imshow('image_rois_with_probs', image_rois_with_probs)
        cv2.waitKey(0)


def verify_iou(rois, probs):
    # probs 按从大到小排序, roi 与后面的所有 roi 的最大 iou 是否小于 overlap_thresh
    roi_idxs = np.argsort(probs)[::-1]
    for idx, roi_idx in enumerate(roi_idxs):
        roi = rois[roi_idx]
        other_rois = rois[roi_idxs[idx + 1:]]
        x1_intersection = np.maximum(roi[0], other_rois[:, 0])
        y1_intersection = np.maximum(roi[1], other_rois[:, 1])
        x2_intersection = np.minimum(roi[2], other_rois[:, 2])
        y2_intersection = np.minimum(roi[3], other_rois[:, 3])
        w_intersection = np.maximum(0, x2_intersection - x1_intersection)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection)
        # shape 为 (last,)
        area_intersection = w_intersection * h_intersection
        # 计算最大 prob 的 box 和其他所有 box 的并集
        area_union = (roi[2] - roi[0]) * (roi[3] - roi[1]) + (other_rois[:, 2] - other_rois[:, 0]) * (
                other_rois[:, 3] - other_rois[:, 1]) - area_intersection
        # compute the ratio of overlap, 就是 IOU
        # shape 为 (last,)
        overlap = area_intersection / (area_union + 1e-6)
        print('max={}'.format(np.max(overlap)))


def show_roi_with_class_before_rectify(image, rois, y_class, ious, class_id_name_mapping):
    ious = np.array(ious)
    sorted_idxs = np.argsort(ious)[::-1]
    sorted_ious = ious[sorted_idxs]
    sorted_rois = rois[sorted_idxs]
    sorted_y_class = y_class[sorted_idxs]
    pos_idxs = np.where(sorted_ious > 0.5)
    pos_ious = sorted_ious[pos_idxs]
    pos_rois = sorted_rois[pos_idxs]
    pos_y_class = sorted_y_class[pos_idxs]
    colors = [np.random.randint(0, 255, 3) for i in range(len(class_idx_name_mapping))]
    for idx in range(len(pos_ious)):
        roi = pos_rois[idx]
        x1, y1, w, h = roi * C.rpn_stride
        x1 = int(round(x1))
        x2 = int(round(x1 + w))
        y1 = int(round(y1))
        y2 = int(round(y1 + h))
        image_roi_with_class = image.copy()
        class_id = np.argmax(pos_y_class[idx])
        class_name = class_idx_name_mapping[class_id]
        cv2.rectangle(image_roi_with_class, (x1, y1), (x2, y2), colors[class_id].tolist(), 2)
        cv2.putText(image_roi_with_class, '{} {:.2f}'.format(class_name, pos_ious[idx]), (x1, y1 + 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255), 1)
        cv2.namedWindow('image_rois_with_class', cv2.WINDOW_NORMAL)
        cv2.imshow('image_rois_with_class', image_roi_with_class)
        cv2.waitKey(0)


def show_roi_with_class(image, rois, y_class, y_regr, ious, class_idx_name_mapping):
    ious = np.array(ious)
    sorted_idxs = np.argsort(ious)[::-1]
    sorted_ious = ious[sorted_idxs]
    sorted_rois = rois[sorted_idxs]
    sorted_y_class = y_class[sorted_idxs]
    sorted_y_regr = y_regr[sorted_idxs]
    pos_idxs = np.where(sorted_ious > 0.5)
    pos_ious = sorted_ious[pos_idxs]
    pos_rois = sorted_rois[pos_idxs]
    pos_y_class = sorted_y_class[pos_idxs]
    pos_y_regr = sorted_y_regr[pos_idxs]
    colors = [np.random.randint(0, 255, 3) for i in range(len(class_idx_name_mapping))]
    for idx in range(len(pos_ious)):
        class_id = np.argmax(pos_y_class[idx])
        class_name = class_idx_name_mapping[class_id]
        roi = pos_rois[idx]
        x1, y1, w, h = roi
        # before rectify
        b_x1, b_y1, b_w, b_h = roi * C.rpn_stride
        b_x1 = int(round(b_x1))
        b_x2 = int(round(b_x1 + b_w))
        b_y1 = int(round(b_y1))
        b_y2 = int(round(b_y1 + b_h))
        b_image = image.copy()
        cv2.rectangle(b_image, (b_x1, b_y1), (b_x2, b_y2), colors[class_id].tolist(), 2)
        cv2.putText(b_image, '{} {:.2f}'.format(class_name, pos_ious[idx]), (x1, y1 + 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255), 1)
        cv2.namedWindow('b_image', cv2.WINDOW_NORMAL)
        cv2.imshow('b_image', b_image)
        # after rectify
        a_image = image.copy()
        y_regr_valid = pos_y_regr[idx][4 * class_id: 4 * (class_id + 1)]
        y_regr = pos_y_regr[idx][4 * (20 + class_id): 4 * (20 + class_id + 1)]
        tx, ty, tw, th = y_regr
        tx /= C.classifier_regr_std[0]
        ty /= C.classifier_regr_std[1]
        tw /= C.classifier_regr_std[2]
        th /= C.classifier_regr_std[3]
        x1, y1, w, h = apply_regr(x1, y1, w, h, tx, ty, tw, th)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(a_image, (x1 * C.rpn_stride, y1 * C.rpn_stride),
                      (x2 * C.rpn_stride, y2 * C.rpn_stride), colors[class_id].tolist(), 2)
        cv2.putText(a_image, '{} {:.2f}'.format(class_name, pos_ious[idx]), (x1, y1 + 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255), 1)
        cv2.namedWindow('a_image', cv2.WINDOW_NORMAL)
        cv2.imshow('a_image', a_image)
        cv2.waitKey(0)


image_input_shape = (None, None, 3)
image_input = Input(shape=image_input_shape)
num_anchors = 9
# define the base network (resnet here, can be VGG, Inception, etc)
base_net_output = nn.base_net(image_input)
rpn_output = nn.rpn(base_net_output, num_anchors)
with open('../config.pickle', 'rb') as f_in:
    C = pickle.load(f_in)
class_name_idx_mapping = C.class_name_idx_mapping
if 'bg' not in class_name_idx_mapping:
    class_name_idx_mapping['bg'] = len(class_name_idx_mapping)
class_idx_name_mapping = {v: k for k, v in class_name_idx_mapping.items()}
model_rpn = Model(image_input, rpn_output)
model_rpn.load_weights('../frcnn_resnet50_8_0.7191_0.1186_0.2485_0.1342_0.9118.hdf5', by_name=True)
annotations = json.load(open('annotation_data.json'))
train_annotations = [annotation for annotation in annotations if annotation['imageset'] == 'train']
image_paths = [annotation['filepath'] for annotation in train_annotations]
for idx, (image_path, train_annotation) in enumerate(zip(image_paths, train_annotations)):
    logger.debug('image_path={}'.format(image_path))
    # image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)
    image_input, ratio, src_image = format_image(image, C)
    # get the feature maps and output from the RPN
    # Y11: rpn_class Y12: rpn_regr
    rpn_class, rpn_regr = model_rpn.predict(image_input)
    rois, probs = rpn_to_roi(rpn_class, rpn_regr, C, overlap_thresh=0.7)
    # show_rois_with_probs(src_image, rois, probs)
    # verify_iou(rois, probs)
    X, Y21, Y22, IoUs = calc_iou(rois, train_annotation, C, class_name_idx_mapping)
    # show_roi_with_class_before_rectify(src_image, X[0], Y21[0], IoUs, class_idx_name_mapping)
    show_roi_with_class(src_image, X[0], Y21[0], Y22[0], IoUs, class_idx_name_mapping)
    print('xx')
