from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
from centernet.detectors.ctdet import CtdetDetector
from boxutils import add_bbox

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def add_2d_detection(img, dets, expected_cls, show_txt=True, center_thresh=0.5):
    right_label = 0
    wrong_labels = 0
    for cat in dets:
        for i in range(len(dets[cat])):
            if dets[cat][i, -1] > center_thresh:
                if cat - 1 == expected_cls:
                    right_label += 1
                else:
                    wrong_labels += 1
                bbox = dets[cat][i, :4]
                add_bbox(img,
                         [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                         cat - 1,
                         ['ov', 'mif'],
                         dets[cat][i, -1],
                         show_txt=show_txt)
    if right_label > 0 and wrong_labels == 0:
        score = 1
    elif right_label == 0 and wrong_labels > 0:
        score = 2
    else:
        score = 0
        if right_label + wrong_labels > 0:
            score = 3
    return img, score


if __name__ == '__main__':
    opt = DotDict({'demo': '/media/palm/data/MicroAlgae/16_8_62/images',
                   'gpus': [0],
                   'device': None,
                   'arch': 'res_50',
                   'heads': {'hm': 2, 'wh': 2, 'reg': 2},
                   'head_conv': 64,
                   'load_model': '/home/palm/PycharmProjects/algea/snapshots/centernet/algea_dla_800_r50_2/model_best.pth',
                   'test_scales': [1.0],
                   'mean': [0.408, 0.447, 0.470],
                   'std': [0.289, 0.274, 0.278],
                   'debugger_theme': 'dark',
                   'debug': 0,
                   'down_ratio': 4,
                   'pad': 31,
                   'fix_res': True,
                   'input_h': 800,
                   'input_w': 800,
                   'flip': 0.5,
                   'flip_test': False,
                   'nms': False,
                   'num_classes': 2,
                   'reg_offset': True,
                   'K': 100,
                   })
    cls = ['ov', 'mif']
    flags = ['not_found', 'correct', 'wrong', 'mixed']
    detector = CtdetDetector(opt)
    val_set = [s.split(',')[0] for s in
               open('/home/palm/PycharmProjects/algea/dataset/test_annotations').read().split('\n')]
    prd = len(os.listdir('/home/palm/PycharmProjects/algea/dataset/centernet_testset'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset', str(prd)))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset', str(prd), 'correct'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset', str(prd), 'wrong'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset', str(prd), 'not_found'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset', str(prd), 'mixed'))
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            if os.path.join(opt.demo, file_name) in val_set:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    for (image_name) in image_names:
        # if '124' not in image_name:
        #     continue
        ret = detector.run(image_name)

        img = cv2.imread(image_name)

        img, score = add_2d_detection(img, ret['results'], 'mif' in image_name.lower())
        cv2.imwrite(os.path.join('/home/palm/PycharmProjects/algea/dataset/centernet_testset/', str(prd), flags[score], os.path.basename(image_name)), img)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
