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

def nms(results, iou_threshold=0.4):
    '''
    Args:
        results: [num_boxes, Dict]
            Dict: left, top, right, bottom, name, color, confidence
        outputs: [K, Dict]
            K: the number of valid boxes
            Dict: left, top, right, bottom, name, color, confidence
    '''

    outputs = []
    while len(results) > 0:
        ix = np.argmax([result['confidence'] for result in results])
        result = results[ix]
        outputs.append(result)
        del results[ix]

        box1 = [result['left'], result['top'], result['right'], result['bottom']]
        to_delete = []
        for jx in range(len(results)):
            box2 = [results[jx]['left'], results[jx]['top'], results[jx]['right'], results[jx]['bottom']]
            iou = calc_iou(box1, box2)
            if iou >= iou_threshold:
                to_delete.append(jx)
        for jx in to_delete[::-1]:
            del results[jx]

    return outputs

def soft_nms(results, threshold, sigma=0.5):
    '''
    Args:
        results: [num_boxes, Dict]
            Dict: left, top, right, bottom, name, color, confidence
        outputs: [K, Dict]
            K: the number of valid boxes
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
