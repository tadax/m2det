import numpy as np
import cv2

from mscoco import table

def get_classes(index):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    np.random.seed(420)
    colors = np.random.randint(0, 224, size=(len(classes), 3))
    return classes[index], tuple(colors[index].tolist())
