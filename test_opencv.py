import cv2
import numpy as np

image = np.ones((300, 300, 3), np.uint8)
image = 255 * image
image = image.astype(np.uint8)
cv2.rectangle(image, (0,0), (299, 299), (0, 0, 0), 2)
cv2.rectangle(image, (50, 50), (250, 200), (0, 255, 0), 2)
text = 'hello world'
(retval, baseLine) = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
text_origin = (50, 50)
cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] + baseLine - 5),
              (text_origin[0] + retval[0] + 5, text_origin[1] - retval[1] - 5), (0, 0, 0), 2)
cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] + baseLine - 5),
              (text_origin[0] + retval[0] + 5, text_origin[1] - retval[1] - 5), (255, 255, 255), -1)
cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
cv2.imshow('image', image)
cv2.waitKey(0)
