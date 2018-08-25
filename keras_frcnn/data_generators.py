from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools
from log import logger


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


def get_new_image_size(width, height, image_min_side=800):
    """
    按照 image_min_size 的标准来计算 resized 后的 image size
    :param width:
    :param height:
    :param image_min_side:
    :return:
    """
    if width <= height:
        ratio = float(image_min_side) / width
        resized_height = int(ratio * height)
        resized_width = image_min_side
    else:
        ratio = float(image_min_side) / height
        resized_width = int(ratio * width)
        resized_height = image_min_side

    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [class_name for class_name in class_count.keys() if class_count[class_name] > 0]
        # 对 iterable 中的元素反复执行循环,返回迭代器
        self.class_cycle = itertools.cycle(self.classes)
        self.current_class = next(self.class_cycle)

    def skip_sample_to_balance_classes(self, annotation_data):
        """
        是否 skip 该 annotation_data
        我觉得该方法会大大减少 sample 的数量
        :param annotation_data:
        :return: True|False
        """
        class_in_image = False
        for bbox in annotation_data['bboxes']:
            class_name = bbox['class']
            if class_name == self.curr_class:
                # 直到遇到包含当前 class 的图片才不 skip
                class_in_image = True
                self.curr_class = next(self.class_cycle)
                break
        return not class_in_image


def calc_rpn(C, annotation_data_aug, width, height, resized_width, resized_height, get_feature_map_size):
    # get anchor config
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture
    (output_width, output_height) = get_feature_map_size(resized_width, resized_height)

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

                    # anchor_type indicates whether an anchor should be a target
                    anchor_type = 'negative'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    # 当前 anchor 能获得的最大 iou
                    best_iou_for_current_anchor = 0.0

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
                            # bbox 中心点位置和 anchor 中心点位置的差值 / anchor 的宽
                            tx = (cx - cxa) / (anchor_x2 - anchor_x1)
                            ty = (cy - cya) / (anchor_y2 - anchor_y1)
                            # bbox 宽度的 log 值 / anchor 的宽
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
                                anchor_type = 'positive'
                                num_anchors_for_bbox[bbox_idx] += 1
                                # We update the regression layer target
                                # if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_current_anchor:
                                    best_iou_for_current_anchor = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if anchor_type == 'negative':
                                    anchor_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if anchor_type == 'negative':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                    elif anchor_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 0
                    elif anchor_type == 'positive':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + num_anchor_ratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_bboxes):
        # 如果该 bbox 没有 positive 的 anchor
        if num_anchors_for_bbox[idx] == 0:
            # 如果 bbox 没有 overlap 的 anchor, 无能为力
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            # 否则把最好的非 positive 的 anchor 作为 positive
            else:
                best_anchor_jy = best_anchor_for_bbox[idx, 0]
                best_anchor_ix = best_anchor_for_bbox[idx, 1]
                best_anchor_ratio_idx = best_anchor_for_bbox[idx, 2]
                best_anchor_size_idx = best_anchor_for_bbox[idx, 3]
            y_is_box_valid[
                best_anchor_jy, best_anchor_ix, best_anchor_ratio_idx + num_anchor_ratios * best_anchor_size_idx] = 1
            y_rpn_overlap[
                best_anchor_jy, best_anchor_ix, best_anchor_ratio_idx + num_anchor_ratios * best_anchor_size_idx] = 1
            start = 4 * (best_anchor_ratio_idx + num_anchor_ratios * best_anchor_size_idx)
            y_rpn_regr[best_anchor_jy, best_anchor_ix, start:start + 4] = best_dx_for_bbox[idx, :]

    # y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    # y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    # y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    # pos_locs 为 3 个数组组成的 tuple, where 函数里每一个为 True 的数可以用三维坐标表示,每一维坐标分别落在 tuple 对应的数组里
    # 每一个数组的长度都相同
    positive_anchors = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    negative_anchors = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
    # 通过第一个数组的长度就可以知道 positive anchor 的个数
    num_positive_anchors = len(positive_anchors[0])
    num_negative_anchors = len(negative_anchors[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if num_positive_anchors > num_regions // 2:
        # 随机生成 num_positive_anchors - num_regions // 2 个 下标, 忽略这些下标对应的 anchor
        ignore_positive_anchor_idx = random.sample(range(num_positive_anchors), num_positive_anchors - num_regions // 2)
        y_is_box_valid[
            0, positive_anchors[0][ignore_positive_anchor_idx], positive_anchors[1][ignore_positive_anchor_idx],
            positive_anchors[2][ignore_positive_anchor_idx]] = 0
        num_positive_anchors = num_regions // 2
    if num_negative_anchors + num_positive_anchors > num_regions:
        ignore_negative_anchor_idx = random.sample(range(num_negative_anchors),
                                                   num_negative_anchors - num_positive_anchors)
        y_is_box_valid[
            0, negative_anchors[0][ignore_negative_anchor_idx], negative_anchors[1][ignore_negative_anchor_idx],
            negative_anchors[2][ignore_negative_anchor_idx]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3)

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


def get_anchor_gt(all_annotation_data, class_count, C, get_feature_map_size, mode='train'):
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_annotation_data)
        for annotation_data in all_annotation_data:
            try:
                # 是否需要平衡 sample 的数量,是否忽略此 sample
                if C.balanced_classes and sample_selector.skip_sample_to_balance_classes(annotation_data):
                    continue
                # read in image, and optionally add augmentation
                if mode == 'train':
                    aug_annotation_data, image = data_augment.augment(annotation_data, C, augment=True)
                else:
                    aug_annotation_data, image = data_augment.augment(annotation_data, C, augment=False)

                height, width = image.shape[:2]

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_image_size(width, height, C.image_min_side)

                # resize the image so that smallest side is 800px
                image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, aug_annotation_data, width, height,
                                                     resized_width,
                                                     resized_height,
                                                     get_feature_map_size)
                except Exception as e:
                    logger.exception(e)
                    continue

                # BGR 变成 RGB, 因为 vgg 训练时接收的就是 RGB
                image = image[:, :, (2, 1, 0)]
                image = image.astype(np.float32)
                # Zero-center by mean pixel, and preprocess image
                image[:, :, 0] -= C.image_channel_mean[0]
                image[:, :, 1] -= C.image_channel_mean[1]
                image[:, :, 2] -= C.image_channel_mean[2]
                # ???
                image /= C.img_scaling_factor
                image = np.expand_dims(image, axis=0)
                # 规整因子,什么意思?
                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling
                yield np.copy(image), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], aug_annotation_data

            except Exception as e:
                logger.error(e)
                continue
