from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from frcnn import roi_helpers
from log import logger
import json

parser = OptionParser()
parser.add_option("-p", "--path", dest="test_path", help="Path to test images directory.")
parser.add_option("-m", "--model_path", dest="model_path", help="Path to frcnn model.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_path", dest="config_path", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:  # if images dir  is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

if not options.model_path:  # if model path is not given
    parser.error('Error: path to frcnn model must be specified. Pass --path to command line')

config_path = options.config_path

with open(config_path, 'rb') as f_in:
    C = pickle.load(f_in)

C.network = options.network
if C.network == 'resnet50':
    import frcnn.resnet as nn
elif C.network == 'vgg':
    import frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rotate = False

images_dir = options.test_path


def format_image_size(image, C):
    """ formats the image size based on config """
    image_min_size = float(C.image_min_size)
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
    image_input = format_image_channels(image, C)
    return image_input, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 * 1.0 / ratio))
    real_y1 = int(round(y1 * 1.0 / ratio))
    real_x2 = int(round(x2 * 1.0 / ratio))
    real_y2 = int(round(y2 * 1.0 / ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def show_batch_rois(image, rois, ratio):
    for roi in rois:
        roi[2] += roi[0]
        roi[3] += roi[1]
        roi = roi * 16
        x1, y1, x2, y2 = get_real_coordinates(ratio, roi[0], roi[1], roi[2], roi[3])
        src_image = image.copy()
        cv2.rectangle(src_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('image', src_image)
        cv2.waitKey(0)


def show_not_bg_rois(image, not_bg_class_names, not_bg_class_prob, not_bg_rois, ratio):
    for roi, class_name, class_prob in zip(not_bg_rois, not_bg_class_names, not_bg_class_probs):
        roi[2] += roi[0]
        roi[3] += roi[1]
        roi = roi * 16
        x1, y1, x2, y2 = get_real_coordinates(ratio, roi[0], roi[1], roi[2], roi[3])
        src_image = image.copy()
        cv2.rectangle(src_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = '{}: {}'.format(class_name, int(100 * class_prob))
        # size[0][0] 表示 width, size[0][1] 表示 height, size[1] 表示 baseline
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        text_origin = (x1, y1 + size[0][1])
        cv2.rectangle(src_image, (text_origin[0] - 5, y1 - 5),
                      (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (0, 0, 0), 2)
        cv2.rectangle(src_image, (text_origin[0] - 5, y1 - 5),
                      (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (255, 255, 255), -1)
        cv2.putText(src_image, text, text_origin, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', src_image)
        cv2.waitKey(0)


class_name_idx_mapping = C.class_name_idx_mapping

if 'bg' not in class_name_idx_mapping:
    class_name_idx_mapping['bg'] = len(class_name_idx_mapping)

class_idx_name_mapping = {v: k for k, v in class_name_idx_mapping.items()}
logger.debug('class_idx_name_mapping={}'.format(class_idx_name_mapping))
class_name_color_mapping = {class_idx_name_mapping[idx]: np.random.randint(0, 255, 3) for idx in class_idx_name_mapping}
C.num_rois = int(options.num_rois)
C.model_path = options.model_path

image_input_shape = (None, None, 3)
image_input = Input(shape=image_input_shape)
rois_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
base_net_output = nn.base_net(image_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_scales) * len(C.anchor_ratios)
rpn_output = nn.rpn(base_net_output, num_anchors)
rcnn_output = nn.rcnn(base_net_output, rois_input, C.num_rois, num_classes=len(class_name_idx_mapping))
model_rpn = Model(image_input, rpn_output)
model_rpn.summary()
model_rcnn = Model([image_input, rois_input], rcnn_output)
model_rcnn.summary()

logger.info('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_rcnn.load_weights(C.model_path, by_name=True)
model_rpn.compile(optimizer='sgd', loss='mse')
model_rcnn.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.7

visualise = True

# for test
annotations = json.load(open('annotation_data.json'))
image_paths = [annotation['filepath'] for annotation in annotations if annotation['imageset'] == 'train']
for idx, image_path in enumerate(image_paths):
    # for idx, image_file in enumerate(sorted(os.listdir(images_dir))):
    #     if not image_file.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         continue
    #     logger.debug('image_file={}'.format(image_file))
    logger.debug('image_path={}'.format(image_path))
    start = time.time()
    # image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)
    image_input, ratio = format_image(image, C)

    # get the feature maps and output from the RPN
    # Y11: rpn_class Y12: rpn_regr
    rpn_class, rpn_regr = model_rpn.predict(image_input)

    rois = roi_helpers.rpn_to_roi(rpn_class, rpn_regr, C, overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    rois[:, 2] -= rois[:, 0]
    rois[:, 3] -= rois[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    # 把所有的 num_rois 分成 n 份, 每份 C.num_rois 个
    for batch_idx in range(rois.shape[0] // C.num_rois + 1):
        # show_batch_rois(image, rois[C.num_rois * batch_idx:C.num_rois * (batch_idx + 1), :], ratio)
        batch_rois = np.expand_dims(rois[C.num_rois * batch_idx:C.num_rois * (batch_idx + 1), :], axis=0)
        # 正好整除
        if batch_rois.shape[1] == 0:
            break
        # 不整除, 最后一个 batch 的 roi 数量不足 num_rois, 需要补充一些
        if batch_idx == rois.shape[0] // C.num_rois:
            # pad batch_rois
            curr_shape = batch_rois.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            padded_batch_rois = np.zeros(target_shape).astype(batch_rois.dtype)
            padded_batch_rois[:, :curr_shape[1], :] = batch_rois
            # UNCLEAR: 为什么用第一个 roi 来进行填充
            padded_batch_rois[0, curr_shape[1]:, :] = batch_rois[0, 0, :]
            batch_rois = padded_batch_rois

        # shape: (1,C.num_rois,num_classes) (1,C.num_rois,(num_classes-1) * 4)
        rcnn_class, rcnn_regr = model_rcnn.predict([image_input, batch_rois])
        not_bg_ids = np.where(np.argmax(rcnn_class[0, :, :], axis=-1) != 20)[0]
        not_bg_class_ids = np.argmax(rcnn_class[0, :, :], axis=-1)[not_bg_ids]
        not_bg_class_names = [class_idx_name_mapping[id] for id in not_bg_class_ids]
        not_bg_class_probs = [class_probs[class_id] for class_probs,class_id in zip(rcnn_class[0][not_bg_ids], not_bg_class_ids)]
        not_bg_rois = rois[C.num_rois * batch_idx:C.num_rois * (batch_idx + 1), :][not_bg_ids]
        logger.debug('not_bg_ids={}'.format(not_bg_ids))
        # show_not_bg_rois(image, not_bg_class_names, not_bg_class_probs, not_bg_rois, ratio)
        for roi_idx in range(rcnn_class.shape[1]):
            # roi 分类的最大 prob < bbox_threshold 或者 roi 分类的最大 prob 是 'bg'
            if np.max(rcnn_class[0, roi_idx, :]) < bbox_threshold or np.argmax(rcnn_class[0, roi_idx, :]) == (
                    rcnn_class.shape[2] - 1):
                continue

            class_name = class_idx_name_mapping[np.argmax(rcnn_class[0, roi_idx, :])]

            if class_name not in bboxes:
                bboxes[class_name] = []
                probs[class_name] = []

            (x, y, w, h) = batch_rois[0, roi_idx, :]

            class_id = np.argmax(rcnn_class[0, roi_idx, :])
            try:
                (tx, ty, tw, th) = rcnn_regr[0, roi_idx, 4 * class_id:4 * (class_id + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[class_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[class_name].append(np.max(rcnn_class[0, roi_idx, :]))

    all_detections = []

    for class_name in bboxes:
        class_bboxes = np.array(bboxes[class_name])
        filtered_class_boxes, filtered_class_probs = roi_helpers.non_max_suppression_fast(class_bboxes,
                                                                                          np.array(probs[class_name]),
                                                                                          overlap_thresh=0.5)
        for idx in range(filtered_class_boxes.shape[0]):
            (x1, y1, x2, y2) = filtered_class_boxes[idx, :]
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(image, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_name_color_mapping[class_name][0]), int(class_name_color_mapping[class_name][1]),
                           int(class_name_color_mapping[class_name][2])), 2)

            text = '{}: {}'.format(class_name, int(100 * filtered_class_probs[idx]))
            all_detections.append((class_name, 100 * filtered_class_probs[idx]))
            # size[0][0] 表示 width, size[0][1] 表示 height, size[1] 表示 baseline
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            text_origin = (real_x1, real_y1 + size[0][1])

            cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] - size[0][1] - 5),
                          (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (0, 0, 0), 2)
            cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] - size[0][1] - 5),
                          (text_origin[0] + size[0][0] + 5, text_origin[1] + 5), (255, 255, 255), -1)
            cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    logger.debug('Elapsed time = {}'.format(time.time() - start))
    logger.debug('all_detections={}'.format(all_detections))
    cv2.imshow('image', image)
    cv2.waitKey(0)
