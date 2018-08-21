import os
import cv2
import xml.etree.ElementTree as ET
import os.path as osp


def get_annotation_data(dataset_dir, visualize=False):
    # [{'filepath':'','width':'','height':'','imageset':'trainval|test','bbox':[{'class':'','x1':'','y1':'','x2':'','x3':'','difficulty':''}]}]
    all_annotation_data = []
    # class objects statistics
    classes_count = {}
    # start at 0
    class_name_idx_mapping = {}
    sub_dataset_dirs = [os.path.join(dataset_dir, sub_dataset) for sub_dataset in ['VOC2007', 'VOC2012']]

    print('Parsing annotation files...')
    for sub_dataset_dir in sub_dataset_dirs:
        if not osp.exists(sub_dataset_dir):
            print('sub_dataset_dir={} does not exist'.format(sub_dataset_dir))
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
            print(e)

        annotation_paths = [os.path.join(annotations_dir, annotation_file) for annotation_file in
                            os.listdir(annotations_dir)]
        for annotation_path in annotation_paths:
            try:
                et = ET.parse(annotation_path)
                element = et.getroot()
                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(images_dir, element_filename),
                                       'width': element_width,
                                       'height': element_height,
                                       'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'train'
                    else:
                        annotation_data['imageset'] = 'val'

                    for element_obj in element_objs:
                        class_name = element_obj.find('name').text
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1

                        if class_name not in class_name_idx_mapping:
                            class_name_idx_mapping[class_name] = len(class_name_idx_mapping)

                        obj_bbox = element_obj.find('bndbox')
                        x1 = int(round(float(obj_bbox.find('xmin').text)))
                        y1 = int(round(float(obj_bbox.find('ymin').text)))
                        x2 = int(round(float(obj_bbox.find('xmax').text)))
                        y2 = int(round(float(obj_bbox.find('ymax').text)))
                        # True or False
                        difficulty = int(element_obj.find('difficult').text) == 1
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
                print(e)
                continue
    return all_annotation_data, classes_count, class_name_idx_mapping
