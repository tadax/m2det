import os
import glob
import cv2
import numpy as np
import argparse

from utils.assign_boxes import assign_boxes
from utils.generate_priors import generate_priors
from utils.augment import augment
from mscoco import table

obj = [v for k, v in table.mscoco2017.items()]
sorted(obj, key=lambda x:x[0])
classes = [j for i, j in obj]
colors = np.random.randint(0, 224, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
text_color = (255, 255, 255)
border_size = 3
font_size = 0.4
font_scale = 1

def draw(img, clsid, left, top, right, bottom):
    name = classes[clsid]
    color = tuple(colors[clsid].tolist())
    cv2.rectangle(img, (left, top), (right, bottom), color, border_size)
    (label_width, label_height), baseline = cv2.getTextSize(name, font, font_size, font_scale)
    cv2.rectangle(img, (left, top), (left + label_width, top + label_height), color, -1)
    cv2.putText(img, name, (left, top + label_height - border_size), font, font_size, text_color, font_scale, line_type)

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
    num_classes = 100 # dummpy
    priors = generate_priors()
    print(len(priors))

    paths = []
    label_paths = glob.glob(os.path.join(args.label_dir, '*.txt'))
    label_paths.sort()
    for bb_path in label_paths:
        im_path = os.path.join(args.image_dir, os.path.splitext(os.path.basename(bb_path))[0] + '.jpg')
        if os.path.exists(im_path):
            paths.append([im_path, bb_path])
            
    for im_path, bb_path in paths:
        img = cv2.imread(im_path)
        img_h, img_w = img.shape[:2]

        with open(bb_path) as f:
            lines = f.read().splitlines()
        labels = []
        for line in lines:
            ix, xmin, ymin, xmax, ymax = line.split('\t') 
            onehot_label = np.eye(num_classes)[int(ix)]
            label = [float(xmin), float(ymin), float(xmax), float(ymax)] + onehot_label.tolist()
            labels.append(label)
            
        img0 = cv2.resize(img, (args.image_size, args.image_size))

        for label in labels:
            xmin, ymin, xmax, ymax = [int(float(f) * args.image_size) for f in label[:4]]
            ix = np.argmax(label[4:])
            clsid = int(ix)
            draw(img0, clsid, xmin, ymin, xmax, ymax)

        if args.augmentation:
            img, labels = augment(img, labels, args.image_size)
            if len(labels) == 0:
                continue
            img = np.array(img * 128.0 + 127.5, dtype=np.uint8)
        else:
            img = cv2.resize(img, (args.image_size, args.image_size))

        labels = np.array(labels)

        imgs = []
        num_assignment = []
        for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75]:
            img_ = img.copy()
            y_true = assign_boxes(labels, priors, num_classes, threshold)
            preds = y_true[:, 4:-1]
            boxes = y_true[:, :4]
            objectness = y_true[:, -1]

            decode_bbox = decode_box(boxes, priors)

            for box, pred, obj in zip(decode_bbox, preds, objectness):
                clsid = np.argmax(pred)
                if clsid == 0:
                    # in the case of background
                    continue
                xmin, ymin, xmax, ymax = [int(f * args.image_size) for f in box]
                clsid -= 1 # decrement to skip background class

                left = int(xmin)
                top = int(ymin)
                right = int(xmax)
                bottom = int(ymax)
                draw(img_, clsid, left, top, right, bottom)

            imgs.append(img_)
            num_assignment.append(sum(y_true[:, -1]))

        print(im_path, num_assignment)
        out = np.concatenate([img0] + imgs, axis=1)
        cv2.imshow('', out)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--augmentation', action='store_true', default=False)
    main(parser.parse_args())
