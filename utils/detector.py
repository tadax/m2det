import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse

from m2det import M2Det
from utils.get_prior import get_priors

class Detector:
    def __init__(self, model_path, input_size, num_classes, prob_threshold, nms_threshold):
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.priors = get_priors(input_size=self.input_size)
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        self.net = M2Det(self.inputs, tf.constant(False), self.num_classes)
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

    def nms(self, boxes):
        prob = boxes[:, 0]
        left = boxes[:, 1]
        top = boxes[:, 2]
        right = boxes[:, 3]
        bottom = boxes[:, 4]
        area = (right - left) * (bottom - top)
        I = np.argsort(prob)
        pick = np.zeros_like(prob, dtype=np.int32)
        counter = 0
        while I.size > 0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[:-1]
            x1 = np.maximum(left[i], left[idx])
            y1 = np.maximum(top[i], top[idx])
            x2 = np.minimum(right[i], right[idx])
            y2 = np.minimum(bottom[i], bottom[idx])
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
            inter = w * h
            o = inter / np.maximum((area[i] + area[idx] - inter), 1e-5)
            I = I[np.where(o <= self.nms_threshold)]
        pick = pick[:counter]
        return pick

    def detect(self, img):
        h, w = img.shape[:2]
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (self.input_size, self.input_size))
        inp = inp - [123.68, 116.78, 103.94] # vgg preprocessing

        # shape of y_pred: (?, num_boxes, 4 + num_classes)
        outs = self.sess.run(self.net.prediction, feed_dict={self.inputs: np.array([inp])})[0]
        boxes = outs[:, :4]
        preds = outs[:, 4:]
        decoded_boxes = self.decode_boxes(boxes)

        boxes = []
        detected = []
        for box, pred in zip(decoded_boxes, preds):
            xmin, ymin, xmax, ymax = box
            clsid = np.argmax(pred)
            if clsid == 0:
                # in the case of background
                continue
            clsid -= 1 # decrement to skip background class
            prob = np.max(pred)
            if prob < self.prob_threshold:
                continue
            left = xmin * w
            top = ymin * h
            right = xmax * w
            bottom = ymax * h
            boxes.append([prob, left, top, right, bottom])
            detected.append(clsid)

        results = []
        if len(boxes) > 0:
            boxes = np.array(boxes)
            pick = self.nms(boxes)
            for i in pick:
                box = boxes[i]
                index = detected[i]
                result = [float(box[0]), index, int(box[1]), int(box[2]), int(box[3]), int(box[4])]
                results.append(result)
        return results
