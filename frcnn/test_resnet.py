from frcnn.resnet import get_feature_map_size

size = get_feature_map_size(1024, 1024)
print(size)