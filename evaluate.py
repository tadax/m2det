import os
import glob
import cv2
import numpy as np
import tqdm
import argparse

from utils.detector import Detector
from mscoco import table

def get_class_name(index):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    return classes[index]

def get_class_index(name):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    return classes.index(name)

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

def calc_precision(predict_labels, true_labels, clsid, prob_threshold, iou_threshold):
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

            max_iou = -1
            max_idx = None
            for true_idx, true_el in enumerate(true_label):
                if true_assign[true_idx]:
                    continue
                true_clsid, true_xmin, true_ymin, true_xmax, true_ymax = true_el
                if true_clsid != clsid:
                    true_assign[true_idx] = None
                    continue
                true_box = np.array([true_xmin, true_ymin, true_xmax, true_ymax])
                iou  = calc_iou(pred_box, true_box)
                if iou >= iou_threshold and iou > max_iou:
                    max_iou = iou
                    max_idx = true_idx

            if max_idx is not None:
                predict_assign[pred_idx] = True
                true_assign[max_idx] = True

        tp += sum([1 for e in predict_assign if e == True])
        fp += sum([1 for e in predict_assign if e == False])
        possible_pos += sum([1 for true_el in true_label if int(true_el[0]) == clsid])

    if (tp + fp == 0):
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if possible_pos == 0:
        recall = None
    else:
        recall = tp / possible_pos

    return precision, recall


def main(args):
    det = Detector(
        model_path=args.model_path, 
        input_size=args.input_size, 
        num_classes=args.num_classes, 
        use_sfam=args.sfam,
        threshold=0.05)

    img_paths = glob.glob(os.path.join(args.image_dir, '*'))
    img_paths.sort()

    predict_labels = []
    true_labels = []

    for img_path in tqdm.tqdm(img_paths):
        label_path = os.path.join(args.label_dir, 
                                  os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            lines = f.read().splitlines()

        label = []
        for line in lines:
            elements = line.split('\t')
            class_index = int(elements[0])
            xmin = float(elements[1])
            ymin = float(elements[2])
            xmax = float(elements[3])
            ymax = float(elements[4])
            label.append([class_index, xmin, ymin, xmax, ymax])

        true_labels.append(label)

        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]
        results = det.detect(img)
        
        label = []
        for res in results:
            confidence = res['confidence']
            class_index = get_class_index(res['name'])
            xmin = res['left'] / w_img
            ymin = res['top'] / h_img
            xmax = res['right'] / w_img
            ymax = res['bottom'] / h_img
            label.append([confidence, class_index, xmin, ymin, xmax, ymax])
        predict_labels.append(label)

    print('data size: {}'.format(len(predict_labels)))

    AP = {}
    iou_threshold = 0.50
    for clsid in range(args.num_classes):
        precisions = []
        recalls = []
        for prob_threshold in np.arange(0.0, 1.01, 0.1):
            precision, recall = calc_precision(predict_labels, true_labels, 
                                               clsid, prob_threshold,
                                               iou_threshold)
            if recall is not None:
                precisions.append(precision)
                recalls.append(recall)

        if len(recalls) == 0:
            continue

        step_num = 10
        maximum_precision = np.zeros(step_num + 1)
        for jx in range(len(recalls)):
            v = precisions[jx]
            k = int(recalls[jx] * step_num) # horizontal axis 
            maximum_precision[k] = max(maximum_precision[k], v)
            
        # intrepolation
        v = 0
        for jx in range(step_num + 1):
            v = max(v, maximum_precision[-jx-1])
            maximum_precision[-jx-1] = v
            
        AP[clsid] = np.mean(maximum_precision)

    for class_index, elem in AP.items():
        class_name = get_class_name(class_index)
        print('{} {}: {}'.format(class_index + 1, class_name, elem * 100))

    print('----------')
    print('mAP@0.5: {}'.format(np.mean([v for i, v in AP.items()]) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--sfam', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
