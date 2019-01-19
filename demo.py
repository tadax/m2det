import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse

from utils.detector import Detector
from mscoco import table

def draw(frame, results):
    border_size = 6
    font_size = 0.8
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_color = (255, 255, 255)
    for index, values in results.items():
        for value in values:
            coord, prob = value
            left, top, right, bottom = [int(i) for i in coord]
            name, color = get_classes(index)
            cv2.rectangle(frame, (left, top), (right, bottom), color, border_size)
            (label_width, label_height), baseline = cv2.getTextSize(name, font, font_size, font_scale)
            cv2.rectangle(frame, (left, top), (left + label_width, top + label_height), color, -1)
            cv2.putText(frame, name, (left, top + label_height - border_size), 
                        font, font_size, text_color, font_scale, line_type)
            print('{}: {} - left: {}, top: {}, right: {}, bottom: {}'.format(name, prob, left, top, right, bottom))

def get_classes(index):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    np.random.seed(420)
    colors = np.random.randint(64, 196, size=(len(classes), 3))
    return classes[index], tuple(colors[index].tolist())

def main(args):
    det = Detector(
        model_path=args.model_path, 
        input_size=args.input_size, 
        num_classes=args.num_classes, 
        threshold=args.threshold)

    if args.inputs.endswith('.mp4'):
        cap = cv2.VideoCapture(args.inputs)
        while True:
            ret, img = cap.read()
            if not ret: break
            results = det.detect(img)
            draw(img, results)
            cv2.imshow('', img)
            cv2.waitKey(1)
    else:
        img = cv2.imread(args.inputs)
        results = det.detect(img)
        draw(img, results)
        cv2.imshow('', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--threshold', type=float, default=0.20)
    parser.add_argument('--gpu', type=str, default='0', required=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
