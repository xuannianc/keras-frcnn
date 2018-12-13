import numpy as np
import cv2
from frcnn import data_generators, data_augmentor
import json
import pickle
from frcnn.data_generators_for_test import calc_rpn
from frcnn.roi_helpers import apply_regr


def show_augmented_annotations(image, augmented_annotation):
    print('num_gt_bboxes={}'.format(len(augmented_annotation['bboxes'])))
    for bbox in augmented_annotation['bboxes']:
        text = bbox['class']
        x1 = bbox['x1']
        y1 = bbox['y1']
        x2 = bbox['x2']
        y2 = bbox['y2']
        # size[0][0] 表示 width, size[0][1] 表示 height, size[1] 表示 baseline
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        text_origin = (x1, y1)

        cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] - size[0][1] - 5),
                      (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (0, 0, 0), 2)
        cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] - size[0][1] - 5),
                      (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (255, 255, 255), -1)
        cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.namedWindow('image_with_annotations', cv2.WINDOW_NORMAL)
    cv2.imshow('image_with_annotations', image)
    cv2.waitKey(0)


def show_pos_anchors_before_rectify(image, y_rpn_cls, max_iou_for_anchor, best_bbox_for_anchor):
    row_ids, col_ids, anchor_ids = np.where(np.logical_and(y_rpn_cls[0, :, :, :9] == 1, y_rpn_cls[0, :, :, 9:] == 1))
    print('num_pos_anchors={}'.format(len(row_ids)))
    colors = [np.random.randint(0, 255, 3) for row_id in row_ids]
    for idx in range(len(row_ids)):
        center_x = col_ids[idx] * C.rpn_stride
        center_y = row_ids[idx] * C.rpn_stride
        w = anchors[anchor_ids[idx]][0]
        h = anchors[anchor_ids[idx]][1]
        x1 = int(round(center_x - w // 2))
        x2 = int(round(center_x + w // 2))
        y1 = int(round(center_y - h // 2))
        y2 = int(round(center_y + h // 2))
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[idx].tolist(), 2)
        max_iou = max_iou_for_anchor[row_ids[idx], col_ids[idx], anchor_ids[idx]]
        best_bbox = best_bbox_for_anchor[row_ids[idx], col_ids[idx], anchor_ids[idx]]
        cv2.putText(image, '{:.2f}-{}'.format(max_iou, best_bbox), (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    cv2.namedWindow('image_before_rectify', cv2.WINDOW_NORMAL)
    cv2.imshow('image_before_rectify', image)
    cv2.waitKey(0)


def show_pos_anchors_after_rectify(image, y_rpn_cls, y_rpn_regr):
    row_ids, col_ids, anchor_ids = np.where(np.logical_and(y_rpn_cls[0, :, :, :9] == 1, y_rpn_cls[0, :, :, 9:] == 1))
    for idx in range(len(row_ids)):
        row_id = row_ids[idx]
        col_id = col_ids[idx]
        anchor_id = anchor_ids[idx]
        center_x = col_id
        center_y = row_id
        w = anchors[anchor_id][0] / C.rpn_stride
        h = anchors[anchor_id][1] / C.rpn_stride
        x1 = center_x - int(round(w // 2))
        y1 = center_y - int(round(h // 2))
        tx, ty, tw, th = y_rpn_regr[0, row_id, col_id, 36 + anchor_id * 4: 36 + (anchor_id + 1) * 4]
        x, y, w, h = np.array(apply_regr(x1, y1, w, h, tx, ty, tw, th)) * C.rpn_stride
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow('image_after_rectify', cv2.WINDOW_NORMAL)
    cv2.imshow('image_after_rectify', image)
    cv2.waitKey(0)


def show_neg_anchors(image, y_rpn_cls, max_iou_for_anchor, best_bbox_for_anchor):
    row_ids, col_ids, anchor_ids = np.where(np.logical_and(y_rpn_cls[0, :, :, :9] == 1, y_rpn_cls[0, :, :, 9:] == 0))
    print('num_neg_anchors={}'.format(len(row_ids)))
    colors = [np.random.randint(0, 255, 3) for row_id in row_ids]
    for idx in range(len(row_ids)):
        center_x = col_ids[idx] * C.rpn_stride
        center_y = row_ids[idx] * C.rpn_stride
        w = anchors[anchor_ids[idx]][0]
        h = anchors[anchor_ids[idx]][1]
        x1 = int(round(center_x - w // 2))
        x2 = int(round(center_x + w // 2))
        y1 = int(round(center_y - h // 2))
        y2 = int(round(center_y + h // 2))
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[idx].tolist(), 2)
        max_iou = max_iou_for_anchor[row_ids[idx], col_ids[idx], anchor_ids[idx]]
        best_bbox = best_bbox_for_anchor[row_ids[idx], col_ids[idx], anchor_ids[idx]]
        cv2.putText(image, '{:.2f}-{}'.format(max_iou, best_bbox), (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    cv2.namedWindow('image_neg_anchors', cv2.WINDOW_NORMAL)
    cv2.imshow('image_neg_anchors', image)
    cv2.waitKey(0)


all_annotations = json.load(open('annotation_data.json'))
classes_count = json.load(open('classes_count.json'))
train_annotations = [annotation for annotation in all_annotations if annotation['imageset'] == 'train']
with open('../config.pickle', 'rb') as f_in:
    C = pickle.load(f_in)
if C.network == 'resnet50':
    import frcnn.resnet as nn
elif C.network == 'vgg':
    import frcnn.vgg as nn

anchors = [scale * np.array(ratio) for scale in C.anchor_scales for ratio in C.anchor_ratios]
for annotation in train_annotations:
    augmented_annotation, image = data_augmentor.augment(annotation, C, augment=True)
    # show_augmented_annotations(image.copy(), augmented_annotation)
    height, width = image.shape[:2]
    # get image dimensions for resizing
    # 按照最小的边为 600 进行 resize
    (resized_width, resized_height) = data_generators.get_new_image_size(width, height, C.image_min_size)
    # resize the image so that smallest side is 600px
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    y_rpn_cls, y_rpn_regr, max_iou_for_anchor, best_bbox_for_anchor = calc_rpn(C, annotation, width, height,
                                     resized_width,
                                     resized_height,
                                     nn.get_feature_map_size, image)
    show_pos_anchors_before_rectify(image.copy(), y_rpn_cls, max_iou_for_anchor, best_bbox_for_anchor)
    show_pos_anchors_after_rectify(image.copy(), y_rpn_cls, y_rpn_regr)
    show_neg_anchors(image, y_rpn_cls, max_iou_for_anchor, best_bbox_for_anchor)
