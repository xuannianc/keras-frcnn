import numpy as np
import pdb
import math
from . import data_generators
import copy
from log import logger


def calc_iou(rois, annotation_data_ang, C, class_name_idx_mapping):
    bboxes = annotation_data_ang['bboxes']
    (width, height) = (annotation_data_ang['width'], annotation_data_ang['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = data_generators.get_new_image_size(width, height, C.image_min_side)
    # ground truth bbox 在 feature map 上的坐标
    gtb = np.zeros((len(bboxes), 4))
    width_ratio = resized_height / float(width)
    height_ratio = resized_height / float(height)
    for bbox_idx, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gtb[bbox_idx, 0] = int(round(bbox['x1'] * width_ratio / C.rpn_stride))
        gtb[bbox_idx, 1] = int(round(bbox['x2'] * width_ratio / C.rpn_stride))
        gtb[bbox_idx, 2] = int(round(bbox['y1'] * height_ratio / C.rpn_stride))
        gtb[bbox_idx, 3] = int(round(bbox['y2'] * height_ratio / C.rpn_stride))

    x_roi = []
    # 分类模型的 y,是 one-hot 形式,(None,len(class_name_idx_mapping))
    y_class = []
    # 回归模型的 y,(None,len(class_name_idx_mapping - 1) * 4)
    y_regr = []
    # 标记 regr 是否参与 loss 计算
    y_regr_valid = []
    IoUs = []  # for debugging only

    for ix in range(rois.shape[0]):
        (x1, y1, x2, y2) = rois[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox_idx = -1
        # roi 和每一个 bbox 计算 iou, 取 best_iou
        for bbox_idx in range(len(bboxes)):
            current_iou = data_generators.iou([gtb[bbox_idx, 0], gtb[bbox_idx, 2], gtb[bbox_idx, 1], gtb[bbox_idx, 3]],
                                              [x1, y1, x2, y2])
            if current_iou > best_iou:
                best_iou = current_iou
                best_bbox_idx = bbox_idx

        if best_iou < C.rcnn_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.rcnn_min_overlap <= best_iou < C.rcnn_max_overlap:
                # hard negative example
                class_name = 'bg'
            elif C.rcnn_max_overlap <= best_iou:
                class_name = bboxes[best_bbox_idx]['class']
                # ground truth bbox 的中心点坐标
                gcx = (gtb[best_bbox_idx, 0] + gtb[best_bbox_idx, 1]) / 2.0
                gcy = (gtb[best_bbox_idx, 2] + gtb[best_bbox_idx, 3]) / 2.0
                gw = gtb[best_bbox_idx, 1] - gtb[best_bbox_idx, 0]
                gh = (gtb[best_bbox_idx, 3] - gtb[best_bbox_idx, 2])
                # proposal bbox 的中心点坐标
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                # 计算梯度
                tx = (gcx - cx) / float(w)
                ty = (gcy - cy) / float(h)
                tw = np.log(gw / float(w))
                th = np.log(gh / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_idx = class_name_idx_mapping[class_name]
        # 构建分类模型的 y
        class_label = len(class_name_idx_mapping) * [0]
        class_label[class_idx] = 1
        y_class.append(copy.deepcopy(class_label))
        regr_label = [0] * 4 * (len(class_name_idx_mapping) - 1)
        regr_label_valid = [0] * 4 * (len(class_name_idx_mapping) - 1)
        if class_name != 'bg':
            regr_pos = 4 * class_idx
            sx, sy, sw, sh = C.classifier_regr_std
            regr_label[regr_pos:4 + regr_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            regr_label_valid[regr_pos:4 + regr_pos] = [1, 1, 1, 1]
            y_regr.append(copy.deepcopy(regr_label))
            y_regr_valid.append(copy.deepcopy(regr_label_valid))
        else:
            y_regr.append(copy.deepcopy(regr_label))
            y_regr_valid.append(copy.deepcopy(regr_label_valid))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y_class = np.array(y_class)
    Y_regr = np.concatenate([np.array(y_regr_valid), np.array(y_regr)], axis=1)
    # expand_dims 是为了 batch 那一维度
    return np.expand_dims(X, axis=0), np.expand_dims(Y_class, axis=0), np.expand_dims(Y_regr, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        logger.error(e)
        return x, y, w, h


def apply_regr_np(X, T):
    """
    根据回归梯度进行校正
    :param X: anchor 的坐标, shape: m * n * 4
    :param T: regr 梯度, shape: m * n * 4
    :return: 修正后的 anchor 的坐标 array([x1,y1,x2,y2]...)
    """
    try:
        height, width = X.shape[:2]
        # anchor 的坐标
        xa = X[:, :, 0]
        ya = X[:, :, 1]
        wa = X[:, :, 2]
        ha = X[:, :, 3]
        # regr 梯度
        tx = T[:, :, 0]
        ty = T[:, :, 1]
        tw = T[:, :, 2]
        th = T[:, :, 3]
        # anchor 中心点坐标
        cxa = xa + wa / 2.
        cya = ya + ha / 2.
        # tx = (cxg - cxa) / wa
        # groud truth bbox 的中心点 x,y
        cxg = tx * wa + cxa
        cyg = ty * ha + cya
        # tw = log(wg / wa)
        wg = np.exp(tw.astype(np.float64)) * wa
        hg = np.exp(th.astype(np.float64)) * ha
        xg = cxg - wg / 2.
        yg = cyg - hg / 2.

        xg = np.round(xg)
        yg = np.round(yg)
        wg = np.round(wg)
        hg = np.round(hg)

        xg = xg.reshape(height, width, 1)
        yg = yg.reshape(height, width, 1)
        wg = wg.reshape(height, width, 1)
        hg = hg.reshape(height, width, 1)
        rectified_anchor_cordinates = np.concatenate((xg, yg, wg, hg), axis=-1)
        return rectified_anchor_cordinates
    except Exception as e:
        logger.exception(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """

    :param boxes: proposal 的 box 的坐标 [[x1,y1,x2,y2]...]
    :param probs: box 包含物体的概率 [p1,p2...]
    :param overlap_thresh: 覆盖阈值
    :param max_boxes: 最大返回的 box 个数
    :return:
    """
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.testing.assert_array_less.html
    # 如果两数组的 shape 不同,会抛出异常
    # 如果 shape 相同但是有前一个数组的元素大于后一个数组的元素,也会抛出异常
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # calculate the areas
    # 数组的每个元素分别相乘 [area1,area2...]
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes, 返回的结果是下标数组,每一个元素的值为原数组的元素的下标
    # 如最后一个元素的值表示的是原数组最大值的下标,第一个元素的值表示原数组最小值的下标
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    idxs = np.argsort(probs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        # 最大面积的 box 的下标
        i = idxs[last]
        pick.append(i)

        # 计算最大面积的 box 和其他所有 box 的交集
        # shape (last,)
        x1_intersection = np.maximum(x1[i], x1[idxs[:last]])
        y1_intersection = np.maximum(y1[i], y1[idxs[:last]])
        x2_intersection = np.minimum(x2[i], x2[idxs[:last]])
        y2_intersection = np.minimum(y2[i], y2[idxs[:last]])
        w_intersection = np.maximum(0, x2_intersection - x1_intersection)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection)
        area_intersection = w_intersection * h_intersection
        # 计算最大面积的 box 和其他所有 box 的并集
        area_union = area[i] + area[idxs[:last]] - area_intersection

        # compute the ratio of overlap, 就是 IOU
        overlap = area_intersection / (area_union + 1e-6)

        # delete all indexes from the index list that have
        # 删除掉和当前最大面积的 box 重叠超过阈值的 box
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def rpn_to_roi(rpn_class, rpn_regr, C, max_boxes=300, overlap_thresh=0.9):
    """
    rpn 产生的 anchor 进行修正和筛选, 生成 roi
    :param rpn_class: rpn 分类的结果 (1,m,n,9) m,n 表示 feature map 的高和宽
    :param rpn_regr: rpn 回归的结果 (1,m,n,36)
    :param C: config 对象
    :param use_regr:
    :param max_boxes: 最大 roi 个数
    :param overlap_thresh:
    :return:
    """
    rpn_regr = rpn_regr / C.std_scaling
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_class.shape[0] == 1
    assert rpn_regr.shape[0] == 1
    # feature map 的高,宽和深
    (m, n, d) = rpn_class.shape[1:]

    anchor_idx = 0
    # (m,n,9,4)
    A = np.zeros((m, n, d, 4))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor 在特征图上的宽
            anchor_width = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            # anchor 在特征图上的高
            anchor_height = (anchor_size * anchor_ratio[1]) / C.rpn_stride
            # 该 anchor 的回归梯度
            regr = rpn_regr[0, :, :, 4 * anchor_idx:4 * anchor_idx + 4]
            # X: (m, n) [[0,1...,n-1],[0,1...,n-1]...]
            # Y: (m, n) [[0,0...,0],[1,1...,1],...[m-1,m-1...,m-1]]
            # 其实生成的是一个坐标,特征图上的每个点都有了相应的坐标
            X, Y = np.meshgrid(np.arange(n), np.arange(m))
            # 计算 anchor 的坐标 (x,y,w,h)
            A[:, :, anchor_idx, 0] = X - anchor_width / 2
            A[:, :, anchor_idx, 1] = Y - anchor_height / 2
            A[:, :, anchor_idx, 2] = anchor_width
            A[:, :, anchor_idx, 3] = anchor_height
            # 根据回归梯度修正
            A[:, :, anchor_idx, :] = apply_regr_np(A[:, :, anchor_idx, :], regr)
            # 宽度最小为 1
            A[:, :, anchor_idx, 2] = np.maximum(1, A[:, :, anchor_idx, 2])
            # 高度最小为 1
            A[:, :, anchor_idx, 3] = np.maximum(1, A[:, :, anchor_idx, 3])
            # A[:,:,anchor_idx, 2] 变成了 x2, 即右下角 x 坐标
            A[:, :, anchor_idx, 2] += A[:, :, anchor_idx, 0]
            # A[:,:,anchor_idx, 3] 变成了 y2, 即右下角 y 坐标
            A[:, :, anchor_idx, 3] += A[:, :, anchor_idx, 1]
            # x1 最小为 0
            A[:, :, anchor_idx, 0] = np.maximum(0, A[:, :, anchor_idx, 0])
            # y1 最小为 0
            A[:, :, anchor_idx, 1] = np.maximum(0, A[:, :, anchor_idx, 1])
            # x2 最大为 n - 1
            A[:, :, anchor_idx, 2] = np.minimum(n - 1, A[:, :, anchor_idx, 2])
            # y2 最大为 m - 1
            A[:, :, anchor_idx, 3] = np.minimum(m - 1, A[:, :, anchor_idx, 3])

            anchor_idx += 1

    # (m*n*9,4)
    all_boxes = np.reshape(A, (-1, 4))
    # (m*n*9,1)
    all_probs = rpn_class.reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # 删除不合理的矩形存在, 左上角的坐标不能大于右下角的坐标
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, axis=0)
    all_probs = np.delete(all_probs, idxs, axis=0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result
