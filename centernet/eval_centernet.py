from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
from centernet.detectors.ctdet import CtdetDetector
from yolo.utils import compute_ap, compute_overlap, create_csv_training_instances
import json
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def all_annotation_from_instance(instance):
    all_annotation = [[], []]
    for obj in instance['object']:
        if obj['name'] == 'ov':
            all_annotation[0].append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
        else:
            all_annotation[1].append(np.array([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]))
    return np.array(all_annotation)


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


def evaluate(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    # all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]  # [[bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)], [bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)]]
    # all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    for label in range(num_classes):
        print()
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            print(i, end='\r')
            detections = all_detections[i][label]
            annotations = np.array(all_annotations[i][label])
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
        total_instances.append(num_annotations)

    return average_precisions, total_instances


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

