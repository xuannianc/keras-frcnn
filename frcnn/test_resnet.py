from frcnn.resnet import get_feature_map_size
from keras.applications.resnet50 import preprocess_input
size = get_feature_map_size(1024, 1024)
print(size)