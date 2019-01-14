import os
import glob
import cv2
import argparse

import table

def main(args):
    classes = [None for _ in range(len(table.mscoco2017))]
    for i, v in table.mscoco2017.items():
        classes[v[0]] = v[1]

    paths = glob.glob(os.path.join(args.label_dir, '*.txt'))
    for path in paths:
        with open(path) as f:
            labels = f.read().splitlines()

        ipath = os.path.join(args.image_dir, os.path.splitext(os.path.basename(path))[0] + '.jpg')
        img = cv2.imread(ipath)
        h, w = img.shape[:2]

        for label in labels:
            ix, xmin, ymin, xmax, ymax = label.split('\t')
            print(label)
            cls = classes[int(ix)]
            left = int(float(xmin) * w)
            top = int(float(ymin) * h)
            right = int(float(xmax) * w)
            bottom = int(float(ymax) * h)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(img, cls, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow('', img)
        cv2.waitKey(2000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    main(parser.parse_args())
