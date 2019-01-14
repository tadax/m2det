import argparse
import json
import cv2
import os

import table

def main(args):
    with open(args.annotation_path) as f:
        data = json.load(f)

    annotations = data['annotations']
    for annotation in annotations:
        catid = annotation['category_id']
        clsid = table.mscoco2017[catid][0]
        
        image_filename = '{0:012d}'.format(annotation['image_id']) + '.jpg'
        src = os.path.join(args.image_dir, image_filename)
        if not os.path.exists(src):
            continue

        img = cv2.imread(src)
        h, w = img.shape[:2]
        bbox = annotation['bbox']
        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = (bbox[0] + bbox[2]) / w
        y2 = (bbox[1] + bbox[3]) / h

        label = [str(clsid), str(x1), str(y1), str(x2), str(y2)]

        output_filename = os.path.splitext(image_filename)[0] + '.txt'
        dst = os.path.join(args.output_dir, output_filename)
        with open(dst, 'a') as f:
            f.write('\t'.join(label) + '\n')
        print(label, src)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--annotation_path', required=True)
    parser.add_argument('--output_dir', required=True)
    main(parser.parse_args())
