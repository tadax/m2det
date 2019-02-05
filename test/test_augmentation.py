import os
import glob
import numpy as np
import cv2
import argparse

from utils.augment import augment
from mscoco import table

def main(args):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    colors = np.random.randint(0, 224, size=(len(classes), 3))

    paths = []
    for bb_path in glob.glob(os.path.join(args.label_dir, '*.txt')):
        im_path = os.path.join(args.image_dir, os.path.splitext(os.path.basename(bb_path))[0] + '.jpg')
        if os.path.exists(im_path):
            paths.append([im_path, bb_path])

    for im_path, bb_path in paths:
        npimg = np.fromfile(im_path, dtype=np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        with open(bb_path) as f:
            lines = f.read().splitlines()

        boxes = []
        for line in lines:
            ix, xmin, ymin, xmax, ymax = line.split('\t') 
            onehot_label = np.eye(len(classes))[int(ix)]
            box = [float(xmin), float(ymin), float(xmax), float(ymax)] + onehot_label.tolist()
            boxes.append(box)

        img, boxes = augment(img, boxes, args.image_size)
        img = np.array(img * 128.0 + 127.5, dtype=np.uint8)
        img_h, img_w = img.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        text_color = (255, 255, 255)
        border_size = 3
        font_size = 0.4
        font_scale = 1

        for box in boxes:
            left = int(box[0] * img_w)
            top = int(box[1] * img_h)
            right = int(box[2] * img_w)
            bottom = int(box[3] * img_h)
            cls = np.argmax(box[4:])
            name = classes[cls]
            color = tuple(colors[cls].tolist())
            cv2.rectangle(img, (left, top), (right, bottom), color, border_size)
            (label_width, label_height), baseline = cv2.getTextSize(name, font, font_size, font_scale)
            cv2.rectangle(img, (left, top), (left + label_width, top + label_height), color, -1)
            cv2.putText(img, name, (left, top + label_height - border_size), font, font_size, text_color, font_scale, line_type)
            #print('{} - left: {}, top: {}, right: {}, bottom: {}'.format(name, left, top, right, bottom))

        cv2.imshow('', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--image_size', type=int, default=320)
    main(parser.parse_args())
