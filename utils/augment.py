import numpy as np
import cv2

def normalize(img):
    img = (img - 127.5) / 128.0
    return img

def random_crop(img, boxes):
    if np.random.uniform() > 0.5:
        return img, boxes

    x1, x2, y1, y2 = np.random.uniform(low=0.0, high=0.20, size=4)
    img_h, img_w = img.shape[:2]

    p1 = int(x1 * img_w)
    p2 = int((1.0 - x2) * img_w)
    q1 = int(y1 * img_h)
    q2 = int((1.0 - y2) * img_h)
    img = img[q1:q2, p1:p2, :]

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

def down_sample(img):
    img_h, img_w = img.shape[:2]
    k = max(int(np.random.normal(loc=2.0)), 1)
    if k > 1:
        img = cv2.resize(img, (int(img_w / k), int(img_h / k)), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return img

def multi_scale(img, boxes):
    if np.random.uniform() > 0.5:
        return img, boxes

    img_h, img_w = img.shape[:2]
    margin_left = int(min(max(np.random.normal(), 0.0), 0.5) * img_w)
    margin_right = int(min(max(np.random.normal(), 0.0), 0.5) * img_w)
    margin_top = int(min(max(np.random.normal(loc=0.1), 0.0), 0.5) * img_h)
    margin_bottom = int(min(max(np.random.normal(loc=0.1), 0.0), 0.5) * img_h)
    new_w = img_w + margin_left + margin_right
    new_h = img_h + margin_top + margin_bottom
    x1 = margin_left
    x2 = margin_left + img_w
    y1 = margin_top
    y2 = margin_top + img_h
    out = np.ones((new_h, new_w, 3), dtype=np.uint8) * 127
    out[y1:y2, x1:x2, :] = img

    scaled_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        xmin = ((margin_left + xmin * img_w) / new_w)
        xmax = ((margin_left + xmax * img_w) / new_w)
        ymin = ((margin_top + ymin * img_h) / new_h)
        ymax = ((margin_top + ymax * img_h) / new_h)
        box = [xmin, ymin, xmax, ymax] + box[4:]
        scaled_boxes.append(box)

    return out, scaled_boxes

def scale(img, labels, img_size):
    img_h, img_w = img.shape[:2]
    ratio = max(img_h, img_w) / img_size
    new_h = int(img_h / ratio)
    new_w = int(img_w / ratio)
    ox = (img_size - new_w) // 2
    oy = (img_size - new_h) // 2
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    out = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
    out[oy:oy + new_h, ox:ox + new_w, :] = scaled

    scaled_labels = []
    for label in labels:
        xmin, ymin, xmax, ymax = label[0:4]
        xmin = (xmin * new_w + ox) / img_size
        ymin = (ymin * new_h + oy) / img_size
        xmax = (xmax * new_w + ox) / img_size
        ymax = (ymax * new_h + oy) / img_size
        label = [xmin, ymin, xmax, ymax] + label[4:]
        scaled_labels.append(label)

    return out, scaled_labels

def augment(img, boxes, input_size):
    img, boxes = random_crop(img, boxes)
    img, boxes = random_flip(img, boxes)
    #img, boxes = multi_scale(img, boxes)
    #img = down_sample(img)
    img, boxes = scale(img, boxes, input_size)
    img = normalize(img)
    return img, boxes
