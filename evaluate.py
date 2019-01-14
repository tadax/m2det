import os
import glob
import cv2
import numpy as np
import tqdm
import argparse

from utils.detector import Detector

def calc_iou(box1, box2):
    # box: left, top, right, bottom
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    intersection = w * h
    s1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou

def calc_precision(predict_labels, true_labels, iou_threshold, clsid, prob_threshold):
    tp = 0
    fp = 0
    possible_pos = 0
    for predict_label, true_label in zip(predict_labels, true_labels):
        if len(predict_label) == 0:
            possible_pos += len(true_label)
            continue

        predict_assign = [False for _ in range(len(predict_label))]
        true_assign = [False for _ in range(len(true_label))]
        for pred_idx, pred_el in enumerate(predict_label):
            prob, pred_clsid, pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_el
            if prob < prob_threshold or pred_clsid != clsid:
                predict_assign[pred_idx] = None
                continue
            pred_box = np.array([pred_xmin, pred_ymin, pred_xmax, pred_ymax])
            for true_idx, true_el in enumerate(true_label):
                if true_assign[true_idx]:
                    continue
                true_clsid, true_xmin, true_ymin, true_xmax, true_ymax = true_el
                true_clsid = int(true_clsid)
                true_xmin = float(true_xmin)
                true_ymin = float(true_ymin)
                true_xmax = float(true_xmax)
                true_ymax = float(true_ymax)
                if true_clsid != clsid:
                    true_assign[true_idx] = None
                    continue
                true_box = np.array([true_xmin, true_ymin, true_xmax, true_ymax])
                iou  = calc_iou(pred_box, true_box)
                if iou >= iou_threshold:
                    predict_assign[pred_idx] = True
                    true_assign[true_idx] = True
                    break

        tp += sum([1 for e in predict_assign if e == True])
        fp += sum([1 for e in predict_assign if e == False])
        possible_pos += len(true_label)

    if (tp + fp == 0):
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if possible_pos == 0:
        recall = 0.0
    else:
        recall = tp / possible_pos

    return precision, recall

def main(args):
    det = Detector(
        model_path=args.model_path, 
        input_size=args.input_size, 
        num_classes=args.num_classes, 
        prob_threshold=0.05,
        nms_threshold=0.50)

    img_paths = glob.glob(os.path.join(args.image_dir, '*'))
    img_paths = img_paths[:100]
    predict_labels = []
    true_labels = []

    for img_path in tqdm.tqdm(img_paths):
        label_path = os.path.join(args.label_dir, 
                                  os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            lines = f.read().splitlines()
        labels = [line.split('\t') for line in lines]
        true_labels.append(labels)

        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]
        results = det.detect(img)
        results = sorted(results, key=lambda x:x[0], reverse=True)
        predict_label = []
        for result in results:
            prob, clsid, xmin, ymin, xmax, ymax = result
            xmin /= w_img
            ymin /= h_img
            xmax /= w_img
            ymax /= h_img
            predict_label.append([prob, clsid, xmin, ymin, xmax, ymax])
        predict_labels.append(predict_label)

    print('Eval size: {}'.format(len(predict_labels)))

    APs = []
    step_num = 10
    for iou_threshold in [0.5]:
        for clsid in range(args.num_classes):
            precisions = []
            recalls = []
            for prob_threshold in np.arange(0.0, 1.01, 0.05):
                precision, recall = calc_precision(predict_labels, true_labels, iou_threshold, 
                                                   clsid, prob_threshold)
                precisions.append(precision)
                recalls.append(recall)

            maximum_precision = np.zeros(step_num + 1)
            for idx in range(len(recalls)):
                v = precisions[idx]
                k = int(recalls[idx] * step_num) # horizontal axis 
                maximum_precision[k] = max(maximum_precision[k], v)

            # intrepolation
            v = 0
            for idx in range(step_num + 1):
                v = max(v, maximum_precision[-idx-1])
                maximum_precision[-idx-1] = v

            AP = np.mean(maximum_precision)
            APs.append(AP)

    mAP = np.mean(APs)
    print('mAP: {}'.format(mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--gpu', type=str, default='0', required=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
