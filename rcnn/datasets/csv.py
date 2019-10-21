from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from retinanet.preprocessing.csv_generator import _read_annotations, _open_for_csv, _read_classes
# import PIL
import numpy as np
import csv
import uuid
from .imdb import imdb
import cv2


class csvdb(imdb):
    def __init__(self, csv_path, class_path):
        imdb.__init__(self, f'csv_{"train" if "train" in csv_path else "val"}')
        self.csv_path = csv_path
        self.class_path = class_path
        self._classes = ('__background__',  # always index 0
                         'mif', 'ov')
        self._image_ext = '.jpg'
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self.image_data = []
        self._image_index = [i for i in range(len(self.image_data))]
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        return self.image_paths[index]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        gt_roidb = self.create_csv_training_instances()
        return gt_roidb

    def create_csv_training_instances(self):
        with _open_for_csv(self.class_path) as file:
            classes = _read_classes(csv.reader(file, delimiter=','))
        with _open_for_csv(self.csv_path) as file:
            train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
        self.classes_ = classes
        self.image_data = train_image_data
        self.image_paths = []
        self._image_index = []
        roidb = []
        labels = sorted(list(classes))
        for idx, k in enumerate(train_image_data):
            image_data = train_image_data[k]
            boxes = []
            gt_classes = []
            seg_areas = []
            for i, obj in enumerate(image_data):
                box = np.array([obj['x1'] - 1, obj['x2'] - 1, obj['y1'] - 1, obj['y2'] - 1], dtype='uint16')
                gt_class = labels.index(obj['class'])+1
                seg_area = (obj['x2'] - obj['x1'] + 1) * (obj['y2'] - obj['y1'] + 1)
                boxes.append(box)
                gt_classes.append(gt_class)
                seg_areas.append(seg_area)
            x = cv2.imread(k)
            height, width, _ = x.shape
            self.image_paths.append(k)
            roidb.append({'boxes': np.array(boxes, dtype='uint16'),
                          'gt_classes': np.array(gt_classes, dtype='int32'),
                          'gt_ishard': np.zeros((len(boxes),), dtype='int32'),
                          'gt_overlaps': np.zeros((len(boxes),), dtype='float32'),
                          'flipped': False,
                          'seg_areas': np.array(seg_areas, dtype='float32'),
                          'width': width,
                          'height': height,
                          'image': k,
                          'img_id': idx

                          })

        return roidb

    def selective_search_roidb(self):
        gt_roidb = self.gt_roidb()
        # ss_roidb = self._load_selective_search_roidb(gt_roidb)
        # roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)

        return gt_roidb

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def evaluate_detections(self, all_boxes):
        raise NotImplementedError('What a pain, I\'ll fking do this later')

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = csvdb('/home/palm/PycharmProjects/algea/dataset/train_annotations')
    res = d.roidb
    from IPython import embed

    embed()
