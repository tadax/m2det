import numpy as np

def encode_box(box, priors):
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:], box[2:])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0])
    area_gt *= (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    iou = inter / union

    encoded_box = np.zeros((len(priors), 4))
    assign_mask = iou > 0.5
    encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = priors[assign_mask]
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:])
    assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])

    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] /= 0.1 # variance0
    encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
    encoded_box[:, 2:4][assign_mask] /= 0.2 # variance1
    return encoded_box.ravel()

def assign_boxes(boxes, priors, num_classes):
    num_classes += 1 # add background class
    assignment = np.zeros((len(priors), 4 + num_classes + 1))
    assignment[:, 4] = 1.0 # background
    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4], priors)
    encoded_boxes = encoded_boxes.reshape(-1, len(priors), 4)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
    best_iou_mask = best_iou > 0 # judge by iou between prior and bbox
    best_iou_idx = best_iou_idx[best_iou_mask]
    assign_num = len(best_iou_idx)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    assignment[:, 4][best_iou_mask] = 0
    assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
    assignment[:, -1][best_iou_mask] = 1
    return assignment
