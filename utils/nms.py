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

def naive_nms(boxes, threshold, iou_threshold=0.3, max_instances=20):
    boxes = np.array(boxes)
    classes = boxes[:, 0]
    unique_classes = [int(i) for i in list(set(classes))]

    results = {}
    for cls in unique_classes:
        mask = classes == cls
        mask_boxes = (boxes[:, 1:])[mask]
        probs = mask_boxes[:, 0] # prob
        coords = mask_boxes[:, 1:] # left, top, right, bottom

        mask = probs >= threshold
        coords = coords[mask]
        probs = probs[mask]
        if len(coords) == 0:
            continue
        results[cls] = []
        
        while len(coords) > 0:
            if len(results[cls]) >= max_instances:
                break

            index = np.argmax(probs)
            coord = coords[index]
            prob = probs[index]
            results[cls].append((prob, coord))
            coords = np.delete(coords, index, axis=0)
            probs = np.delete(probs, index, axis=0)

            ious = np.array([calc_iou(coord, c) for c in coords])
            mask = ious < iou_threshold
            coords = coords[mask]
            probs = probs[mask]

    return results

def nms(boxes, threshold, sigma=0.5, max_instances=20):
    '''
    boxes shape: [num_boxes, 6]
    second dimension: class, prob, left, top, right, bottom
    '''

    boxes = np.array(boxes)
    classes = boxes[:, 0]
    unique_classes = [int(i) for i in list(set(classes))]

    results = {}
    for cls in unique_classes:
        mask = classes == cls
        mask_boxes = (boxes[:, 1:])[mask]
        probs = mask_boxes[:, 0] # prob
        coords = mask_boxes[:, 1:] # left, top, right, bottom

        mask = probs >= threshold
        coords = coords[mask]
        probs = probs[mask]
        if len(coords) == 0:
            continue
        results[cls] = []

        while len(coords) > 0:
            if len(results[cls]) >= max_instances:
                break

            index = np.argmax(probs)
            coord = coords[index]
            prob = probs[index]
            results[cls].append((prob, coord))
            coords = np.delete(coords, index, axis=0)
            probs = np.delete(probs, index, axis=0)

            # soft-nms with a Gaussian penalty function
            ious = [calc_iou(coord, c) for c in coords]
            penalty = np.array([np.e ** (-iou ** 2 / sigma) for iou in ious])
            probs = probs * penalty
            mask = probs >= threshold
            coords = coords[mask]
            probs = probs[mask]

    return results
