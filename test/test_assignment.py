import os
import glob
import cv2
import numpy as np
import copy
import argparse

from utils.assign_boxes import assign_boxes
from utils.get_prior import get_priors
from utils.generate_priors import generate_priors
from utils.nms import nms
from mscoco import table

def draw(img, clsid, left, top, right, bottom):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    np.random.seed(420)
    colors = np.random.randint(0, 224, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_color = (255, 255, 255)
    border_size = 2
    name = classes[clsid]
    color = tuple(colors[clsid].tolist())
    cv2.rectangle(img, (left, top), (right, bottom), color, border_size)

def decode_box(boxes, priors):
    prior_width = priors[:, 2] - priors[:, 0]
    prior_height = priors[:, 3] - priors[:, 1]
    prior_center_x = 0.5 * (priors[:, 2] + priors[:, 0])
    prior_center_y = 0.5 * (priors[:, 3] + priors[:, 1])
    decode_bbox_center_x = boxes[:, 0] * 0.1 * prior_width # variance0
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = boxes[:, 1] * 0.1 * prior_height # variance0
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(boxes[:, 2] * 0.2) # variance1
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(boxes[:, 3] * 0.2) # variance1
    decode_bbox_height *= prior_height
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None], decode_bbox_ymin[:, None],
                                  decode_bbox_xmax[:, None], decode_bbox_ymax[:, None]), axis=-1)
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox

def main(args):
    priors = generate_priors(args.image_size)
    #priors = get_priors(args.image_size)

    paths = []
    for bb_path in glob.glob(os.path.join(args.label_dir, '*.txt')):
        im_path = os.path.join(args.image_dir, os.path.splitext(os.path.basename(bb_path))[0] + '.jpg')
        if os.path.exists(im_path):
            paths.append([im_path, bb_path])
            
    for im_path, bb_path in paths:
        print(im_path)
        img = cv2.imread(im_path)
        img_h, img_w = img.shape[:2]

        with open(bb_path) as f:
            lines = f.read().splitlines()
        labels = []
        for line in lines:
            ix, xmin, ymin, xmax, ymax = line.split('\t') 
            onehot_label = np.eye(args.num_classes)[int(ix)]
            label = [float(xmin), float(ymin), float(xmax), float(ymax)] + onehot_label.tolist()
            labels.append(label)
            
        img0 = copy.deepcopy(img)

        for label in labels:
            xmin, ymin, xmax, ymax = label[:4]
            ix = np.argmax(label[4:])
            clsid = int(ix)
            left = int(float(xmin) * img_w)
            top = int(float(ymin) * img_h)
            right = int(float(xmax) * img_w)
            bottom = int(float(ymax) * img_h)
            draw(img0, clsid, left, top, right, bottom)

        labels = np.array(labels)

        y_true = assign_boxes(labels, priors, args.num_classes)
        preds = y_true[:, 4:-1]
        boxes = y_true[:, :4]

        decode_bbox = decode_box(boxes, priors)

        boxes = []
        for box, pred in zip(decode_bbox, preds):
            xmin, ymin, xmax, ymax = box
            clsid = np.argmax(pred)
            if clsid == 0:
                # in the case of background
                continue
            clsid -= 1 # decrement to skip background class
            left = int(xmin * img_w)
            top = int(ymin * img_h)
            right = int(xmax * img_w)
            bottom = int(ymax * img_h)
            boxes.append([clsid, 1.0, left, top, right, bottom])

        if len(boxes) > 0:
            boxes = nms(boxes, iou_thr=0.75)

        for box in boxes:
            clsid = box[0]
            left, top, right, bottom = [int(i) for i in box[2]]
            draw(img, clsid, left, top, right, bottom)

        print(len(labels), len(boxes))
        out = np.concatenate((img0, img), axis=1)
        cv2.imshow('', out)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=80)
    main(parser.parse_args())
