from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from evaluate_util import DotDict, all_annotation_from_instance, evaluate
import os
import cv2
from centernet.detectors.ctdet import CtdetDetector
from yolo.utils import create_csv_training_instances
import json
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def detect(dets, expected_cls, center_thresh=0.5):
    right_label = 0
    wrong_labels = 0
    all_detection = [[], []]
    for cat in dets:
        for i in range(len(dets[cat])):
            if dets[cat][i, -1] > center_thresh:
                if cat - 1 == expected_cls:
                    right_label += 1
                else:
                    wrong_labels += 1
                all_detection[cat-1].append(np.array(dets[cat][i], dtype='uint16'))
    return all_detection


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
    config_path = '/home/palm/PycharmProjects/algea/yolo/algeaconfig.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        config['train']['train_csv'],
        config['valid']['valid_csv'],
        config['train']['classes_csv'],
    )
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
    all_detections = []
    all_annotations = []
    for instance in valid_ints:
        ret = detector.run(instance['filename'])

        img = cv2.imread(instance['filename'])

        all_detection = detect(ret['results'], 'mif' in instance['filename'].lower())
        all_annotation = all_annotation_from_instance(instance)
        all_annotations.append(all_annotation)
        all_detections.append(all_detection)
        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        # break

    average_precisions, total_instances = evaluate(all_detections, all_annotations, 2)
    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
        sum([a * b for a, b in zip(total_instances, average_precisions)]) / sum(total_instances)))
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions) / sum(x > 0 for x in total_instances)))  # mAP: 0.5000
