import cv2
import numpy as np
import copy


def augment(annotation, config, augment=True):
    augmented_annotation = copy.deepcopy(annotation)
    image = cv2.imread(augmented_annotation['filepath'])

    if augment:
        height, width = image.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            # cv2.flip 第二个参数, 0: 竖直翻转 1:水平翻转 -1:竖直水平翻转
            #  __________|___________
            # |          |           |
            # |  _____   | x1,y1__   |
            # | |_____|  |  |_____|  |(width - x2)
            # |__________|_____x2,y2_|
            #            |
            image = cv2.flip(image, 1)
            for bbox in augmented_annotation['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = width - x1
                bbox['x1'] = width - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            #  _________________
            # |x1,y1_           |
            # | |____|          |
            # |       x2,y2     |
            # |                 |
            # |                 |
            # |-----------------|
            # |                 |
            # |                 |
            # |  ____           |
            # | |____|          |
            # |_________________|
            image = cv2.flip(image, 0)
            for bbox in augmented_annotation['bboxes']:
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

            for bbox in augmented_annotation['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 90:
                    #  _____________________         _________
                    # |                     |         _ x1,y1 |
                    # |  _____              |       || |      |
                    # | |_____|             |       ||_|      |
                    # |_____________________|    x2,y2        |
                    #                               |         |
                    #                               |         |
                    #                               |         |
                    #                               |_________|
                    bbox['x1'] = height - y2
                    bbox['x2'] = height - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 180:
                    #  _____________________         ____________________
                    # |                     |       |      x2,y2 _____   |
                    # |  _____              |       |           |_____|  |
                    # | |_____|             |       |               x1,y1|
                    # |_____________________|       |____________________|
                    #
                    bbox['x1'] = width - x2
                    bbox['x2'] = width - x1
                    bbox['y1'] = height - y2
                    bbox['y2'] = height - y1
                elif angle == 270:
                    #  _____________________         _________
                    # |                     |       |         |
                    # |  _____              |       |         |
                    # | |_____|             |       |         |
                    # |_____________________|       |         |
                    #                               |      _  x2,y2
                    #                               |     | | |
                    #                               |     |_| |
                    #                               |x1,y1____|
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = width - x2
                    bbox['y2'] = width - x1
                elif angle == 0:
                    pass

    augmented_annotation['width'] = image.shape[1]
    augmented_annotation['height'] = image.shape[0]
    return augmented_annotation, image
