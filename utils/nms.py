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

def nms(boxes, threshold, iou_threshold=0.4):
    boxes = np.array(boxes)
    classes = boxes[:, 0]
    probs = boxes[:, 1]
    coords = boxes[:, 2:]

    index = np.argsort(probs)[::-1]
    classes = classes[index]
    probs = probs[index]
    coords = coords[index]

    mask = probs >= threshold
    classes = classes[mask]
    probs = probs[mask]
    coords = coords[mask]

    results = {}
    while len(probs) > 0:
        clsid = int(classes[0])
        prob = probs[0]
        coord = coords[0]
        classes = np.delete(classes, 0, axis=0)
        coords = np.delete(coords, 0, axis=0)
        probs = np.delete(probs, 0, axis=0)

        ious = np.array([calc_iou(coord, c) for c in coords])
        mask = ious < iou_threshold
        classes = classes[mask]
        probs = probs[mask]
        coords = coords[mask]

        if clsid in results:
            results[clsid].append((prob, coord))
        else:
            results[clsid] = [(prob, coord)]
        
    return results

def class_nms(boxes, threshold, iou_threshold=0.25, max_instances=20):
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

def soft_nms(results, threshold, sigma=0.5):
    '''
    Args:
        results: [num_boxes, Dict]
            Dict: left, top, right, bottom, name, color, confidence
    '''

    outputs = []
    while len(results) > 0:
        ix = np.argmax([result['confidence'] for result in results])
        result = results[ix]
        outputs.append(result)
        del results[ix]

        # soft-nms with a Gaussian penalty function
        box1 = [result['left'], result['top'], result['right'], result['bottom']]
        to_delete = []
        for jx in range(len(results)):
            box2 = [results[jx]['left'], results[jx]['top'], results[jx]['right'], results[jx]['bottom']]
            iou = calc_iou(box1, box2)
            penalty = np.e ** (-iou ** 2 / sigma)
            results[jx]['confidence'] = results[jx]['confidence'] * penalty
            if results[jx]['confidence'] < threshold:
                to_delete.append(jx)

        for jx in to_delete[::-1]:
            del results[jx]

    return outputs
