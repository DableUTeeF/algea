import base64
import cv2
import numpy as np
from imageio import imread
import io
import os
import json
if __name__ == '__main__':
    path = '/media/palm/data/MicroAlgae/16_8_62/jsn/'
    ipath = '/media/palm/data/MicroAlgae/16_8_62/images/'
    csv = []
    classid = []
    for folder in os.listdir(path):
        # for sub_folder in os.listdir(os.path.join(path, folder)):
            for file in os.listdir(os.path.join(path, folder, )):
                im_json = json.load(open(os.path.join(path, folder, file)))
                impath = os.path.join(ipath, file[:-5]+'.jpg')
                img = imread(io.BytesIO(base64.b64decode(im_json['imageData'])))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.concatenate((np.expand_dims(img, 2), np.expand_dims(img, 2), np.expand_dims(img, 2)), 2)
                cv2.imwrite(impath, img)
                for obj in im_json['shapes']:
                    x = int(obj['points'][0][0]), int(obj['points'][1][0])
                    y = int(obj['points'][0][1]), int(obj['points'][1][1])
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    label = 'ov' if 'ov' in obj["label"].lower() else 'mif'
                    csv.append(f'{impath},{x1},{y1},{x2},{y2},{label}')
                    classid.append(label)

    with open('algea/annotations', 'w') as wr:
        for element in csv:
            wr.write(element)
            wr.write('\n')
    classid = list(set(classid))
    x = 0
    with open('algea/classes', 'w') as wr:
        for i, e in enumerate(classid):
            wr.write(str(e))
            wr.write(str(','))
            wr.write(str(x))
            wr.write(str('\n'))
            x += 1
