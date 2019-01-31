import numpy as np
import cv2

def normalize(img):
    img = (img - 127.5) / 128.0
    return img

def random_crop(img, boxes, input_size, ratio=0.20):
    x1, x2, y1, y2 = np.random.uniform(low=0.0, high=ratio, size=4)
    img_h, img_w = img.shape[:2]

    p1 = int(x1 * img_w)
    p2 = int((1.0 - x2) * img_w)
    q1 = int(y1 * img_h)
    q2 = int((1.0 - y2) * img_h)
    img = img[q1:q2, p1:p2, :]
    img = cv2.resize(img, (input_size, input_size), 
                     interpolation=cv2.INTER_LINEAR)

    cropped_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        if ((x1 >= xmax) or (xmin >= 1.0 - x2) or (y1 >= ymax) or (ymin >= 1.0 - y2)):
            continue
        xmin = max((xmin - x1) / (1.0 - x1 - x2), 0.0)
        xmax = 1.0 - max((1.0 - xmax - x2) / (1.0 - x1 - x2), 0.0)
        ymin = max((ymin - y1) / (1.0 - y1 - y2), 0.0)
        ymax = 1.0 - max((1.0 - ymax - y2) / (1.0 - y1 - y2), 0.0)
        box = [xmin, ymin, xmax, ymax] + box[4:]
        cropped_boxes.append(box)

    return img, cropped_boxes

def random_flip(img, boxes):
    if np.random.uniform() > 0.5:
        img = cv2.flip(img, 1)
        flipped_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box[:4]
            new_xmin = 1.0 - xmax
            new_xmax = 1.0 - xmin
            box = [new_xmin, ymin, new_xmax, ymax] + box[4:]
            flipped_boxes.append(box)
    else:
        flipped_boxes = boxes

    return img, flipped_boxes

def augment(img, boxes, input_size):
    img, boxes = random_crop(img, boxes, input_size)
    img, boxes = random_flip(img, boxes)
    img = normalize(img)
    return img, boxes
