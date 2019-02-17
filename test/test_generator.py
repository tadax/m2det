import cv2
import numpy as np
from utils.generate_priors import generate_priors

def main(image_size=320):
    anchors = generate_priors()
    print(anchors)

    for anchor in anchors:
        ymin, xmin, ymax, xmax = anchor
        left = int(max(xmin, 0) * image_size)
        top = int(max(ymin, 0) * image_size)
        right = int(min(xmax, 1.0) * image_size)
        bottom = int(min(ymax, 1.0) * image_size)
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), -1)
        cv2.imshow('', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
