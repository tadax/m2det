import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse

from utils.detector import Detector
from utils.drawing import draw


def main(args):
    det = Detector(
        model_path=args.model_path, 
        input_size=args.input_size, 
        num_classes=args.num_classes, 
        use_sfam=args.sfam,
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
    elif os.path.isdir(args.inputs):
        paths = glob.glob(os.path.join(args.inputs, '*'))
        for path in paths:
            print(path)
            img = cv2.imread(path)
            results = det.detect(img)
            draw(img, results)
            cv2.imshow('', img)
            cv2.waitKey(0)
    else:
        img = cv2.imread(args.inputs)
        results = det.detect(img)
        draw(img, results)
        cv2.imshow('', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--model_path', default='weights/variables')
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--sfam', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='-1')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
