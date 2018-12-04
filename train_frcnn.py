from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from frcnn import config, data_generators
from frcnn import losses as losses
import frcnn.roi_helpers as roi_helpers
from log import logger
from keras.utils import generic_utils
import json


parser = OptionParser()
# DATASET_DIR = '/home/adam/.keras/datasets/VOCdevkit'
parser.add_option("-d", "--dataset", dest="dataset_dir", help="Path to training dataset.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of 'simple' or 'pascal_voc'",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                  default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg and resnet50.",
                  default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training.",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training.",
                  action="store_true", default=False)
parser.add_option("--rot", "--rotate", dest="rotate",
                  help="Augment with 90 degree rotations in training.",
                  action="store_true", default=False)
parser.add_option("--image_min_size", type="int", dest="image_min_size", help="Min side of image to resize.", default=800)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_output_path", dest="config_output_path",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--model_weight_path", dest="model_weight_path", help="Output path for model weights.")
parser.add_option("--base_net_weight_path", dest="base_net_weight_path",
                  help="Path for base network weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.dataset_dir:  # if dataset_dir is not specified
    parser.error('Path to training dataset must be specified. Pass -d or --dataset to command line')

if options.parser == 'pascal_voc':
    from frcnn.pascal_voc_parser import get_annotation_data
elif options.parser == 'simple':
    from frcnn.simple_parser import get_data
else:
    parser.error("Option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()
C.use_horizontal_flips = options.horizontal_flips
C.use_vertical_flips = options.vertical_flips
C.rotate = options.rotate
C.num_rois = options.num_rois
C.image_min_size = options.image_min_size

if options.network == 'vgg':
    C.network = 'vgg'
    from frcnn import vgg as nn
elif options.network == 'resnet50':
    from frcnn import resnet as nn
    C.network = 'resnet50'
else:
    parser.error("Option network must be one of 'vgg' or 'resnet50'")

# check if output weight path was passed via command line
if options.model_weight_path:
    C.model_weight_path = options.model_weight_path
else:
    C.model_weight_path = 'faster_rcnn_{}_weight.hdf5'.format(C.network)

# check if base weight path was passed via command line
if options.base_net_weight_path:
    C.base_net_weights_path = options.base_net_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights_path = nn.get_weight_path()

# all_annotation_data, classes_count, class_name_idx_mapping = get_annotation_data(DATASET_DIR)
all_annotations = json.load(open('annotation_data.json'))
classes_count = json.load(open('classes_count.json'))
class_name_idx_mapping = json.load(open('class_name_idx_mapping.json'))

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_name_idx_mapping['bg'] = len(class_name_idx_mapping)

C.class_name_idx_mapping = class_name_idx_mapping

logger.debug('class_count={}'.format(classes_count))
logger.info('Num of classes (including bg) = {}'.format(len(classes_count)))
logger.debug('class_name_idx_mapping={}'.format(class_name_idx_mapping))

config_output_path = options.config_output_path

with open(config_output_path, 'wb') as config_f:
    pickle.dump(C, config_f)
    logger.info('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_path))

random.shuffle(all_annotations)

train_annotations = [annotation for annotation in all_annotations if annotation['imageset'] == 'train']
val_annotations = [annotation for annotation in all_annotations if annotation['imageset'] == 'val']

logger.info('Num of samples {}'.format(len(all_annotations)))
logger.info('Num of train samples {}'.format(len(train_annotations)))
logger.info('Num of val samples {}'.format(len(val_annotations)))

train_data_gen = data_generators.get_anchor_gt(train_annotations, classes_count, C, nn.get_feature_map_size, mode='train')
val_data_gen = data_generators.get_anchor_gt(val_annotations, classes_count, C, nn.get_feature_map_size, mode='val')

input_shape = (None, None, 3)
image_input = Input(shape=input_shape)
roi_input = Input(shape=(None, 4))
# define the base network (resnet here, can be VGG, Inception, etc)
base_net_output = nn.base_net(image_input)
# define the RPN, built on the base net
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_output = nn.rpn(base_net_output, num_anchors)
# [(batch_size=1, num_rois, num_classes),(batch_size=1, num_rois, 4 * (num_classes -1))
rcnn_output = nn.rcnn(base_net_output, roi_input, C.num_rois, num_classes=len(classes_count))
model_rpn = Model(image_input, rpn_output)
model_rpn.summary()
model_rcnn = Model([image_input, roi_input], rcnn_output)
model_rcnn.summary()
# this is a model that holds both the RPN and the RCNN, used to load/save weights for the models
model = Model([image_input, roi_input], rpn_output + rcnn_output)

try:
    print('loading weights from {}'.format(C.base_net_weights_path))
    model_rpn.load_weights(C.base_net_weights_path, by_name=True)
    model_rcnn.load_weights(C.base_net_weights_path, by_name=True)
except:
    print('Could not load pretrained model weights of base net. '
          'Weights can be found in https://github.com/fchollet/deep-learning-models/releases')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_class_loss(num_anchors), losses.rpn_regr_loss(num_anchors)])
model_rcnn.compile(optimizer=optimizer_classifier,
                   loss=[losses.rcnn_class_loss, losses.rcnn_regr_loss(len(classes_count) - 1)],
                   metrics={'rcnn_class':'accuracy'})
model.compile(optimizer='sgd', loss='mae')

num_epochs = int(options.num_epochs)
# 每 1000 个 epoch 输出详细日志保存模型
num_iters = 1000
# 迭代的序号
iter_idx = 0
# 每个 epoch 的 positive roi 的个数
num_pos_rois_per_epoch = []
losses = np.zeros((num_iters, 5))
start_time = time.time()
best_loss = np.Inf

print('Starting training...')
for epoch_idx in range(num_epochs):
    progbar = generic_utils.Progbar(num_iters)
    print('Epoch {}/{}'.format(epoch_idx + 1, num_epochs))
    while True:
        try:
            X1, Y1, aug_annotation_data = next(train_data_gen)
            # loss_rpn = [loss,rpn_out_class_loss,rpn_out_regress_loss], 名字的组成有最后一层的 name + '_loss'
            # 这里还要注意 label 的 shape 可以和模型输出的 shape 不一样,这取决于 loss function
            rpn_loss = model_rpn.train_on_batch(X1, Y1)
            # [(1,m,n,9),(1,m,n,36)]
            rpn_prediction = model_rpn.predict_on_batch(X1)
            # (boxes,probs) boxes:(None,4) (x1,y1,x2,y2) probs:(None,1)
            rois = roi_helpers.rpn_to_roi(rpn_prediction[0], rpn_prediction[1], C, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X2: x_roi Y21: y_class Y22: y_regr
            X2, Y21, Y22, IouS = roi_helpers.calc_iou(rois, aug_annotation_data, C, class_name_idx_mapping)

            if X2 is None:
                num_pos_rois_per_epoch.append(0)
                continue
            # 假设 Y21 为 np.array([[[0,0,0,1],[0,0,0,0]]]),np.where(Y21[0,:,-1]==1) 返回 (array([0]),)
            neg_idxes = np.where(Y21[0, :, -1] == 1)
            pos_idxes = np.where(Y21[0, :, -1] == 0)

            if len(neg_idxes) > 0:
                neg_idxes = neg_idxes[0]
            else:
                neg_idxes = []

            if len(pos_idxes) > 0:
                pos_idxes = pos_idxes[0]
            else:
                pos_idxes = []

            num_pos_rois_per_epoch.append((len(pos_idxes)))

            if C.num_rois > 1:
                # 如果正样本个数不足 num_rois//2,全部送入训练
                if len(pos_idxes) < C.num_rois // 2:
                    selected_pos_idxes = pos_idxes.tolist()
                # 如果正样本个数超过 num_rois//2, 随机抽取 num_rois//2 个 送入训练
                else:
                    selected_pos_idxes = np.random.choice(pos_idxes, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_idxes = np.random.choice(neg_idxes, C.num_rois - len(selected_pos_idxes),
                                                          replace=False).tolist()
                except:
                    selected_neg_idxes = np.random.choice(neg_idxes, C.num_rois - len(selected_pos_idxes),
                                                          replace=True).tolist()

                selected_idxes = selected_pos_idxes + selected_neg_idxes
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_idxes = pos_idxes.tolist()
                selected_neg_idxes = neg_idxes.tolist()
                if np.random.randint(0, 2):
                    selected_idxes = random.choice(neg_idxes)
                else:
                    selected_idxes = random.choice(pos_idxes)

            rcnn_loss = model_rcnn.train_on_batch([X1, X2[:, selected_idxes, :]],
                                                  [Y21[:, selected_idxes, :], Y22[:, selected_idxes, :]])

            losses[iter_idx, 0] = rpn_loss[1]
            losses[iter_idx, 1] = rpn_loss[2]
            losses[iter_idx, 2] = rcnn_loss[1]
            losses[iter_idx, 3] = rcnn_loss[2]
            # accuracy
            losses[iter_idx, 4] = rcnn_loss[3]

            iter_idx += 1

            progbar.update(iter_idx,
                           [('rpn_class_loss', np.mean(losses[:iter_idx, 0])),
                            ('rpn_regr_loss', np.mean(losses[:iter_idx, 1])),
                            ('rcnn_class_loss', np.mean(losses[:iter_idx, 2])),
                            ('rcnn_regr_loss', np.mean(losses[:iter_idx, 3]))])

            if iter_idx == num_iters:
                rpn_class_loss = np.mean(losses[:, 0])
                rpn_regr_loss = np.mean(losses[:, 1])
                rcnn_class_loss = np.mean(losses[:, 2])
                rcnn_regr_loss = np.mean(losses[:, 3])
                rcnn_class_acc = np.mean(losses[:, 4])
                mean_num_pos_rois = float(sum(num_pos_rois_per_epoch)) / len(num_pos_rois_per_epoch)
                num_pos_rois_per_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_num_pos_rois))
                    if mean_num_pos_rois == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                    print('RPN Classification Loss: {}'.format(rpn_class_loss))
                    print('RPN Regression Loss : {}'.format(rpn_regr_loss))
                    print('RCNN Classification Loss: {}'.format(rcnn_class_loss))
                    print('RCNN Regression Loss: {}'.format(rcnn_regr_loss))
                    print('RCNN Classification Accuracy: {}'.format(rcnn_class_acc))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = rpn_class_loss + rpn_regr_loss + rcnn_class_loss + rcnn_regr_loss
                iter_idx = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model.save_weights(C.model_weight_path)
                break

        except Exception as e:
            logger.exception('{}'.format(e))
            continue

logger.info('Training complete, exiting.')
