import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse

from m2det import M2Det
from utils.generate_priors import generate_priors
from utils.nms import soft_nms, nms
from utils.classes import get_classes

class Detector:
    def __init__(self, model_path, input_size, num_classes, use_sfam, threshold):
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_sfam = use_sfam
        self.threshold = threshold
        self.priors = generate_priors()
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        self.net = M2Det(self.inputs, tf.constant(False), self.num_classes, self.use_sfam)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def decode_boxes(self, boxes):
        prior_width = self.priors[:, 2] - self.priors[:, 0]
        prior_height = self.priors[:, 3] - self.priors[:, 1]
        prior_center_x = 0.5 * (self.priors[:, 2] + self.priors[:, 0])
        prior_center_y = 0.5 * (self.priors[:, 3] + self.priors[:, 1])
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

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        ratio = max(img_h, img_w) / self.input_size
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.input_size - new_w) // 2
        oy = (self.input_size - new_h) // 2
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        inp = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8) * 127
        inp[oy:oy + new_h, ox:ox + new_w, :] = scaled
        inp = (inp - 127.5) / 128.0
        return inp, ox, oy, new_w, new_h

    def detect(self, img):
        img_h, img_w = img.shape[:2]
        inp, ox, oy, new_w, new_h = self.preprocess(img)

        outs = self.sess.run(self.net.prediction, feed_dict={self.inputs: np.array([inp])})[0]

        boxes = self.decode_boxes(outs[:, :4])
        preds = np.argmax(outs[:, 4:], axis=1)
        confidences = np.max(outs[:, 4:], axis=1)

        # skip background class
        mask = np.where(preds > 0)
        boxes = boxes[mask]
        preds = preds[mask]
        confidences = confidences[mask]

        mask = np.where(confidences >= self.threshold)
        boxes = boxes[mask]
        preds = preds[mask]
        confidences = confidences[mask]

        results = []
        for box, clsid, conf in zip(boxes, preds, confidences):
            xmin, ymin, xmax, ymax = box
            left = int((xmin * self.input_size - ox) / new_w * img_w)
            top = int((ymin * self.input_size - oy) / new_h * img_h)
            right = int((xmax * self.input_size - ox) / new_w * img_w)
            bottom = int((ymax * self.input_size - oy) / new_h * img_h)
            conf = float(conf)
            name, color = get_classes(clsid - 1)
            results.append({
                'left': left,
                'top': top,
                'right': right, 
                'bottom': bottom,
                'name': name,
                'color': color,
                'confidence': conf,
            })

        #results = nms(results)
        results = soft_nms(results, self.threshold)
            
        return results
