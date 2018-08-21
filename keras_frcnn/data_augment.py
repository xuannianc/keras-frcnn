import cv2
import numpy as np
import copy


def augment(annotation_data, config, augment=True):
    annotation_data_aug = copy.deepcopy(annotation_data)
    image = cv2.imread(annotation_data_aug['filepath'])

    if augment:
        height, width = image.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            # 0: 竖直翻转 1:水平翻转 -1:竖直水平翻转
            image = cv2.flip(image, 1)
            for bbox in annotation_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = width - x1
                bbox['x1'] = width - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            image = cv2.flip(image, 0)
            for bbox in annotation_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = height - y1
                bbox['y1'] = height - y2

        if config.rot_90:
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                image = np.transpose(image, (1, 0, 2))
                image = cv2.flip(image, 0)
            elif angle == 180:
                image = cv2.flip(image, -1)
            elif angle == 90:
                image = np.transpose(image, (1, 0, 2))
                image = cv2.flip(image, 1)
            elif angle == 0:
                pass

            for bbox in annotation_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = width - x2
                    bbox['y2'] = width - x1
                elif angle == 180:
                    bbox['x2'] = width - x1
                    bbox['x1'] = width - x2
                    bbox['y2'] = height - y1
                    bbox['y1'] = height - y2
                elif angle == 90:
                    bbox['x1'] = height - y2
                    bbox['x2'] = height - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    annotation_data_aug['width'] = image.shape[1]
    annotation_data_aug['height'] = image.shape[0]
    return annotation_data_aug, image
