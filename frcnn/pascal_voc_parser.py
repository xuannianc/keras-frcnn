import os
import cv2
import xml.etree.ElementTree as ET
import os.path as osp
from log import logger
import json


def get_annotation_data(dataset_dir, visualize=False):
    # [{'filepath':'','width':'','height':'','imageset':'trainval|test','bbox':[{'class':'','x1':'','y1':'','x2':'','x3':'','difficulty':True|False}]}]
    all_annotation_data = []
    # class objects statistics
    classes_count = {}
    # start at 0
    class_name_idx_mapping = {}
    sub_dataset_dirs = [os.path.join(dataset_dir, sub_dataset) for sub_dataset in ['VOC2007', 'VOC2012']]

    logger.info('Parsing annotation files...')
    for sub_dataset_dir in sub_dataset_dirs:
        if not osp.exists(sub_dataset_dir):
            logger.debug('sub_dataset_dir={} does not exist'.format(sub_dataset_dir))
            continue
        annotations_dir = os.path.join(sub_dataset_dir, 'Annotations')
        images_dir = os.path.join(sub_dataset_dir, 'JPEGImages')
        trainval_txt_path = os.path.join(sub_dataset_dir, 'ImageSets', 'Main', 'trainval.txt')
        trainval_files = []
        try:
            with open(trainval_txt_path) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            logger.error(e)

        annotation_paths = [os.path.join(annotations_dir, annotation_file) for annotation_file in
                            os.listdir(annotations_dir)]
        for annotation_path in annotation_paths:
            try:
                et = ET.parse(annotation_path)
                root = et.getroot()
                # <object>
                objects = root.findall('object')
                # <filename>
                filename = root.find('filename').text
                # <width>
                width = int(root.find('size').find('width').text)
                # <height>
                height = int(root.find('size').find('height').text)

                if len(objects) > 0:
                    annotation_data = {'filepath': os.path.join(images_dir, filename),
                                       'width': width,
                                       'height': height,
                                       'bboxes': []}

                    if filename in trainval_files:
                        annotation_data['imageset'] = 'train'
                    else:
                        annotation_data['imageset'] = 'val'

                    for object in objects:
                        class_name = object.find('name').text
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1

                        if class_name not in class_name_idx_mapping:
                            class_name_idx_mapping[class_name] = len(class_name_idx_mapping)

                        obj_bbox = object.find('bndbox')
                        x1 = int(round(float(obj_bbox.find('xmin').text)))
                        y1 = int(round(float(obj_bbox.find('ymin').text)))
                        x2 = int(round(float(obj_bbox.find('xmax').text)))
                        y2 = int(round(float(obj_bbox.find('ymax').text)))
                        # True or False
                        difficulty = int(object.find('difficult').text) == 1
                        annotation_data['bboxes'].append(
                            {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                    all_annotation_data.append(annotation_data)

                    if visualize:
                        image = cv2.imread(annotation_data['filepath'])
                        for bbox in annotation_data['bboxes']:
                            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                        cv2.imshow('image', image)
                        cv2.waitKey(0)
            except Exception as e:
                logger.exception(e)
                continue
    return all_annotation_data, classes_count, class_name_idx_mapping


def serialize():
    dataset_dir = '/home/adam/.keras/datasets/VOCdevkit'
    all_annotation_data, classes_count, class_name_idx_mapping = get_annotation_data(dataset_dir, visualize=True)
    json.dump(all_annotation_data, open('annotation_data.json', 'w'))
    json.dump(classes_count, open('classes_count.json', 'w'))
    json.dump(class_name_idx_mapping, open('class_name_idx_mapping.json', 'w'))


if __name__ == '__main__':
    serialize()
