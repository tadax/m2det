import numpy as np

def calc_iou(box1, box2):
    # box: left, top, right, bottom
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    intersection = w * h
    s1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = max(s1 + s2 - intersection, 1e-5)
    iou = intersection / union
    return iou

def per_class_nms(boxes, iou_thr):
    '''
    boxes shape: [num_boxes, 6]
    second dimension: class, prob, left, top, right, bottom
    '''

    boxes = np.array(boxes)
    classes = boxes[:, 0]
    unique_classes = [int(i) for i in list(set(classes))]

    results = []
    for cls in unique_classes:
        mask = classes == cls
        mask_boxes = (boxes[:, 1:])[mask]
        mask_boxes = mask_boxes[mask_boxes[:, 0].argsort()[::-1]] # sort by prob
        probs = mask_boxes[:, 0] # prob
        coords = mask_boxes[:, 1:] # left, top, right, bottom

        while len(coords) > 0:
            coord = coords[0]
            prob = probs[0]
            results.append((cls, prob, coord))
            coords = coords[1:]
            probs = probs[1:]
            mask = np.array([calc_iou(coord, x) for x in coords]) < iou_thr
            coords = coords[mask]
            probs = probs[mask]

    return results

def standard_nms(boxes, iou_thr):
    '''
    boxes shape: [num_boxes, 6]
    second dimension: class, prob, left, top, right, bottom
    '''

    boxes = np.array(boxes)
    boxes = boxes[boxes[:, 1].argsort()[::-1]]
    classes = boxes[:, 0]
    probs = boxes[:, 1]
    coords = boxes[:, 2:]

    results = []
    while len(coords) > 0:
        cls = classes[0]
        prob = probs[0]
        coord = coords[0]
        results.append((int(cls), prob, coord))
        classes = classes[1:]
        probs = probs[1:]
        coords = coords[1:]
        mask = np.array([calc_iou(coord, x) for x in coords]) < iou_thr
        classes = classes[mask]
        probs = probs[mask]
        coords = coords[mask]

    return results

def soft_nms(boxes, iou_thr):
    # To be implemented
    return

def nms(boxes, iou_thr=0.25):
    return standard_nms(boxes, iou_thr)
