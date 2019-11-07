import base64
import cv2
import numpy as np
from imageio import imread
import io
import os
import json
from PIL import Image

if __name__ == '__main__':
    csv_train_ann = open('/home/palm/PycharmProjects/algea/dataset/train_annotations').read().split('\n')[:-1]
    csv_val_ann = open('/home/palm/PycharmProjects/algea/dataset/test_annotations').read().split('\n')[:-1]

    train_images = list(set([os.path.basename(element.split(',')[0]) for element in csv_train_ann]))
    val_images = list(set([os.path.basename(element.split(',')[0]) for element in csv_val_ann]))

    imout_dir = ['train', 'val']

    path = '/media/palm/data/MicroAlgae/16_8_62/jsn/'
    ipath = '/media/palm/data/MicroAlgae/16_8_62/images'
    csv = []
    classid = []
    train_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    val_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    json_dict = [train_json_dict, val_json_dict]
    bnd_id = [1, 1]  # START_BOUNDING_BOX_ID
    i = 0
    for folder in os.listdir(path):
        # for sub_folder in os.listdir(os.path.join(path, folder)):
            for file in os.listdir(os.path.join(path, folder)):
                i += 1
                print(i, end='\r')
                im_json = json.load(open(os.path.join(path, folder, file)))
                impath = os.path.join(ipath, file[:-5]+'.jpg')
                img = imread(io.BytesIO(base64.b64decode(im_json['imageData'])))
                pil_img = Image.fromarray(img)

                img_id = file[:-5]
                img_info = {
                    'file_name': img_id+'.jpg',
                    'height': pil_img.height,
                    'width': pil_img.width,
                    'id': img_id
                }
                tid = file[:-5]+'.jpg' in val_images
                cv2.imwrite(os.path.join(ipath, imout_dir[tid], img_id+'.jpg'), img)
                json_dict[tid]['images'].append(img_info)
                for obj in im_json['shapes']:
                    x = int(obj['points'][0][0]), int(obj['points'][1][0])
                    y = int(obj['points'][0][1]), int(obj['points'][1][1])
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    label_id = 0 if 'ov' in obj["label"].lower() else 1  # ov: 0, mif: 1
                    o_width = x2 - x1
                    o_height = y2 - y1
                    ann = {
                        'area': o_width * o_height,
                        'iscrowd': 0,
                        'bbox': [x1, y1, o_width, o_height],
                        'category_id': label_id,
                        'ignore': 0,
                        'segmentation': []  # This script is not for segmentation
                    }
                    ann.update({'image_id': img_id, 'id': bnd_id[tid]})
                    json_dict[tid]['annotations'].append(ann)
                    bnd_id[tid] = bnd_id[tid] + 1
    print('bnd_id', bnd_id)
    for label, label_id in [('ov', 0), ('mif', 1)]:
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        json_dict[0]['categories'].append(category_info)
        json_dict[1]['categories'].append(category_info)

    with open('/media/palm/data/MicroAlgae/16_8_62/annotations/train_algea.json', 'w') as f:
        output_json = json.dumps(json_dict[0])
        f.write(output_json)
    with open('/media/palm/data/MicroAlgae/16_8_62/annotations/val_algea.json', 'w') as f:
        output_json = json.dumps(json_dict[1])
        f.write(output_json)
    with open('./train_algea.json', 'w') as f:
        output_json = json.dumps(json_dict[0])
        f.write(output_json)
    with open('./val_algea.json', 'w') as f:
        output_json = json.dumps(json_dict[1])
        f.write(output_json)
