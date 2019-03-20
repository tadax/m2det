import numpy as np
import cv2

def draw(frame, results):
    ratio = max(frame.shape[:2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_color = (255, 255, 255)
    border_size = int(0.005 * ratio)
    font_size = float(0.001 * ratio)
    font_scale = int(0.0015 * ratio)

    for result in results:
        text = '{}: {}'.format(result['name'], np.round(result['confidence'], 2))
        (label_width, label_height), baseline = cv2.getTextSize(text, font, font_size, font_scale)
        cv2.rectangle(img=frame, 
                      pt1=(result['left'], result['top']),
                      pt2=(result['right'], result['bottom']),
                      color=result['color'],
                      thickness=border_size)
        cv2.rectangle(img=frame, 
                      pt1=(result['left'], result['top']),
                      pt2=(result['left'] + label_width + border_size*2, result['top'] + label_height + border_size*2),
                      color=result['color'], 
                      thickness=-1)
        cv2.putText(img=frame, 
                    text=text, 
                    org=(result['left'] + border_size, result['top'] + label_height + border_size),
                    fontFace=font, 
                    fontScale=font_size, 
                    color=text_color,
                    thickness=font_scale,
                    lineType=line_type)
