from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(a, b, area_intersection):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(a, b):
    # a and b should be (x1,y1,x2,y2)
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    # width
    w = min(a[2], b[2]) - x
    # height
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_image_size(width, height, image_min_side=600):
    if width <= height:
        f = float(image_min_side) / width
        resized_height = int(f * height)
        resized_width = image_min_side
    else:
        f = float(image_min_side) / height
        resized_width = int(f * width)
        resized_height = image_min_side

    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [class_name for class_name in class_count.keys() if class_count[class_name] > 0]
        # 对 iterable 中的元素反复执行循环,返回迭代器
        self.class_cycle = itertools.cycle(self.classes)
        self.current_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_name = bbox['class']

            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, annotation_data_aug, width, height, resized_width, resized_height, output_image_size_calc_function):
    # get anchor config
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture
    (output_width, output_height) = output_image_size_calc_function(resized_width, resized_height)

    num_anchor_ratios = len(anchor_ratios)
    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(annotation_data_aug['bboxes'])
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_idx, bbox in enumerate(annotation_data_aug['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_idx, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_idx, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_idx, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_idx, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(num_anchor_ratios):
            anchor_width = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_height = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            # 遍历 feature map 的所有点,分别映射到输入图像的对应矩形的中心点
            for ix in range(output_width):
                # x-coordinates of the current anchor box
                anchor_x1 = downscale * (ix + 0.5) - anchor_width // 2
                anchor_x2 = downscale * (ix + 0.5) + anchor_width // 2

                # ignore boxes that go across image boundaries
                if anchor_x1 < 0 or anchor_x2 > resized_width:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    anchor_y1 = downscale * (jy + 0.5) - anchor_height // 2
                    anchor_y2 = downscale * (jy + 0.5) + anchor_height // 2

                    # ignore boxes that go across image boundaries
                    if anchor_y1 < 0 or anchor_y2 > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    # 不懂哈...
                    best_iou_for_loc = 0.0

                    for bbox_idx in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_idx, 0], gta[bbox_idx, 2], gta[bbox_idx, 1], gta[bbox_idx, 3]],
                                       [anchor_x1, anchor_y1, anchor_x2, anchor_y2])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_idx] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_idx, 0] + gta[bbox_idx, 1]) / 2.0
                            cy = (gta[bbox_idx, 2] + gta[bbox_idx, 3]) / 2.0
                            cxa = (anchor_x1 + anchor_x2) / 2.0
                            cya = (anchor_y1 + anchor_y2) / 2.0
                            tx = (cx - cxa) / (anchor_x2 - anchor_x1)
                            ty = (cy - cya) / (anchor_y2 - anchor_y1)
                            tw = np.log((gta[bbox_idx, 1] - gta[bbox_idx, 0]) / (anchor_x2 - anchor_x1))
                            th = np.log((gta[bbox_idx, 3] - gta[bbox_idx, 2]) / (anchor_y2 - anchor_y1))
                        if annotation_data_aug['bboxes'][bbox_idx]['class'] != 'bg':

                            # Every GT box should be mapped to an anchor box,
                            # so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_idx]:
                                best_anchor_for_bbox[bbox_idx] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_idx] = curr_iou
                                best_x_for_bbox[bbox_idx, :] = [anchor_x1, anchor_x2, anchor_y1, anchor_y2]
                                best_dx_for_bbox[bbox_idx, :] = [tx, ty, tw, th]

                            # We set the anchor to positive if the IOU is >0.7.
                            # It does not matter if there was another better box, it just indicates overlap.
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_idx] += 1
                                # We update the regression layer target
                                # if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + num_anchor_ratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_bboxes):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + num_anchor_ratios * best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + num_anchor_ratios * best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + num_anchor_ratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    # y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    # y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    # y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_anchor_gt(all_annotation_data, class_count, C, output_image_size_calc_function, backend, mode='train'):
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_annotation_data)

        for annotation_data in all_annotation_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(annotation_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    annotation_data_aug, image = data_augment.augment(annotation_data, C, augment=True)
                else:
                    annotation_data_aug, image = data_augment.augment(annotation_data, C, augment=False)

                height, width = image.shape[:2]

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_image_size(width, height, C.image_min_side)

                # resize the image so that smallest side is 600px
                image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, annotation_data_aug, width, height,
                                                     resized_width,
                                                     resized_height,
                                                     output_image_size_calc_function)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)
                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling
                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], annotation_data_aug

            except Exception as e:
                print(e)
                continue
