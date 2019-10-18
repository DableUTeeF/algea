import numpy as np
import os
import xml.etree.ElementTree as ET
import keras
import csv
import cv2
# from keras.optimizers import Optimizer
# from keras import backend as K
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from retinanet.preprocessing.csv_generator import _read_annotations, _open_for_csv, _read_classes


# from keras.legacy import interfaces


class CocoGenerator:
    """ Generate data from the COCO dataset.

    See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
    """

    def __init__(self, json_path, image_dir):
        """ Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.image_dir = image_dir
        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)

        return annotations


def parse_annotation(ann_dir, img_dir, labels=()):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


def parse_voc_annotation(ann_dir, img_dir, labels=()):
    all_imgs = {}
    seen_labels = {}
    max_box_per_image = 0

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text[:-4]
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs[filename] = img
            if len(img['object']) > max_box_per_image:
                max_box_per_image = len(img['object'])

    return all_imgs, seen_labels, max_box_per_image


def create_voc_training_instances(voc_folder):
    # parse annotations of the training set
    ints, labels, max_box_per_image = parse_voc_annotation(os.path.join(voc_folder, 'Annotations'),
                                                           os.path.join(voc_folder, 'JPEGImages'))

    train_txt = open(os.path.join(voc_folder, 'ImageSets/Main/train.txt')).read().split('\n')[:-1]
    val_txt = open(os.path.join(voc_folder, 'ImageSets/Main/val.txt')).read().split('\n')[:-1]

    train_ints = [ints[train] for train in train_txt]
    valid_ints = [ints[val] for val in val_txt]

    # for instance in ints:
    #     filename = os.path.split(instance['filename'])[-1][:-4]
    #     if filename in train_txt:
    #         train_ints.append(instance)
    #     else:
    #         valid_ints.append(instance)

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_csv_training_instances(train_csv, test_csv, class_csv, with_wh=False):
    with _open_for_csv(class_csv) as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    with _open_for_csv(train_csv) as file:
        train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    with _open_for_csv(test_csv) as file:
        test_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    train_ints = []
    valid_ints = []
    labels = list(classes)
    max_box_per_image = 0
    for k in train_image_data:
        image_data = train_image_data[k]
        ints = {'filename': k, 'object': []}
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            if with_wh:
                x = cv2.imread(k)
                height, width, _ = x.shape
                o['width'] = width
                o['height'] = height
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        train_ints.append(ints)

    for k in test_image_data:
        image_data = test_image_data[k]
        ints = {'filename': k, 'object': []}
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            if with_wh:
                x = cv2.imread(k)
                height, width, _ = x.shape
                o['width'] = width
                o['height'] = height
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        valid_ints.append(ints)

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_coco_training_instances(train_json,
                                   val_json,
                                   train_image_dir,
                                   val_image_dir,
                                   with_empty=True
                                   ):
    train_coco = CocoGenerator(train_json, train_image_dir)
    val_coco = CocoGenerator(val_json, val_image_dir)
    assert sorted(val_coco.labels) == sorted(
        train_coco.labels), r"Something's wrong, the labels in val and train seem to not the same"

    labels = {}
    for label in val_coco.labels:
        labels[val_coco.labels[label]] = 0

    max_box_per_image = 0
    train_ints = []
    valid_ints = []
    for image_index in range(len(train_coco.image_ids)):
        ann = train_coco.load_annotations(image_index)
        image_info = train_coco.coco.loadImgs(train_coco.image_ids[image_index])[0]
        impath = os.path.join(train_coco.image_dir, image_info['file_name'])

        instance = {'filename': impath,
                    'object': [],
                    'width': image_info['width'],
                    'height': image_info['height']}
        for j in range(len(ann['labels'])):
            x1 = int(ann['bboxes'][j][0])
            y1 = int(ann['bboxes'][j][1])
            x2 = int(ann['bboxes'][j][2])
            y2 = int(ann['bboxes'][j][3])
            cls = train_coco.labels[ann['labels'][j]]
            obj = {'xmin': x1, 'xmax': x2, 'ymin': y1, 'ymax': y2, 'name': cls}
            instance['object'].append(obj)
        if with_empty or len(instance['object']) > 0:
            train_ints.append(instance)
        if len(instance['object']) > max_box_per_image:
            max_box_per_image = len(instance['object'])

    for image_index in range(len(val_coco.image_ids)):
        ann = val_coco.load_annotations(image_index)
        image_info = val_coco.coco.loadImgs(val_coco.image_ids[image_index])[0]
        impath = os.path.join(val_coco.image_dir, image_info['file_name'])

        instance = {'filename': impath,
                    'object': [],
                    'width': image_info['width'],
                    'height': image_info['height']}
        for j in range(len(ann['labels'])):
            x1 = int(ann['bboxes'][j][0])
            y1 = int(ann['bboxes'][j][1])
            x2 = int(ann['bboxes'][j][2])
            y2 = int(ann['bboxes'][j][3])
            cls = val_coco.labels[ann['labels'][j]]
            obj = {'xmin': x1, 'xmax': x2, 'ymin': y1, 'ymax': y2, 'name': cls}
            instance['object'].append(obj)
        if with_empty or len(instance['object']) > 0:
            valid_ints.append(instance)
        if len(instance['object']) > max_box_per_image:
            max_box_per_image = len(instance['object'])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_training_instances(train_annot_folder,
                              train_image_folder,
                              valid_annot_folder,
                              valid_image_folder,
                              labels,
                              ):
    # parse annotations of the training set
    train_ints, train_labels = parse_annotation(train_annot_folder, train_image_folder, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8 * len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('\033[33m\nThese labels has no image')
            for label in labels:
                if label not in overlap_labels:
                    print(label)
            print('\033[0m')
        labels = list(overlap_labels)
    else:
        print('No labels are provided. Train on all seen labels.')
        # print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape
    color = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0)]
    for box in boxes:
        xmin = max(0, int(box.xmin * image_w))
        ymin = max(0, int(box.ymin * image_h))
        xmax = min(int(box.xmax * image_w), image_w)
        ymax = min(int(box.ymax * image_h), image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color[box.get_label() % 6], 3)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    color[box.get_label() % 6], 1)

    return image


def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def decode_netoutv3(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if objectness <= obj_thresh:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)

            boxes.append(box)

    return boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1. / scale


def _constrain(min_v, max_v, value):
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w) / image_w, float(new_h) / image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin'] * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax'] * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin'] * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax'] * sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    try:
        im_sized = cv2.resize(image, (new_w, new_h))
    except cv2.error as e:
        print('something')
        print(new_w, new_h)
        raise cv2.error('{}, {} {}'.format(new_w, new_h, e.__cause__))
    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(im_sized, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]

    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w, :]


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def label_to_coco_label(label):
    return {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
            16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33,
            29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
            42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60,
            55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77,
            68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}[label]


def coco_label_to_label(coco_label):
    dictionary = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
                  15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
                  27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
                  39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
                  51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
                  63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
                  75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    for label, d_coco_label in dictionary.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if d_coco_label == coco_label:
            return label
    return -1


def boundbox2cocobox(boxes, scale):
    """
    :param scale:
    :param boxes: [Bndbox(), Bndbox(),...]
    :return: boxes: [[x, y, w, h]]
             scores: float
             labels: int
    """
    cocoboxes = []
    scores = []
    labels = []
    for bbox in boxes:
        cocoboxes.append([bbox.xmin / scale,
                          bbox.ymin / scale,
                          (bbox.xmax - bbox.xmin) / scale,
                          (bbox.ymax - bbox.ymin) / scale])
        scores.append(bbox.get_score())
        labels.append(bbox.get_label())

    assert len(cocoboxes) == len(scores) == len(labels)
    return cocoboxes, scores, labels


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


# noinspection PyTypeChecker
def evaluate_coco(generator, model, anchors, json_path, imsize=448, threshold=0.5):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results

    import pickle
    if os.path.exists('coco_eval_temp.pk'):
        results, image_ids = pickle.load(open('coco_eval_temp.pk', 'rb'))

    else:
        results = []
        image_ids = []
        for index in range(generator.size()):
            # if index % 50 == 0:
            #     print()
            print(index, end='\r')

            image = generator.load_image(index)
            image, scale = resize_image(image, 360, imsize)

            image = np.expand_dims(image, 0)
            boxes = get_yolo_boxes(model,
                                   image,
                                   imsize, imsize,
                                   anchors,
                                   0.5,
                                   0.5,
                                   preprocess=True)[0]

            boxes, scores, labels = boundbox2cocobox(boxes, scale)
            # assert len(boxes) > 0
            # compute predicted labels and scores
            image_id = int(os.path.split(generator.instances[index]['filename'])[-1][:-4])
            for box, score, label in zip(boxes, scores, labels):
                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                image_result = {
                    'image_id': image_id,
                    'category_id': label_to_coco_label(label),  # todo:
                    'score': float(score),
                    'bbox': box,
                }

                # append detection to results
                results.append(image_result)

            # append image to list of processed images
            image_ids.append(image_id)
    with open('coco_eval_temp.pk', 'wb') as wr:
        pickle.dump([results, image_ids], wr)
    if not len(results):
        return
    import json
    # write output
    json.dump(results, open('{}_bbox_results.json'.format('val2017'), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format('val2017'), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = COCO(json_path)
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format('val2017'))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


# noinspection PyTypeChecker
def evaluate(model,
             generator,
             iou_threshold=0.5,
             obj_thresh=0.5,
             nms_thresh=0.45,
             net_h=416,
             net_w=416,
             save_path=None):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        obj_thresh      : The threshold used to distinguish between object and non-object
        nms_thresh      : The threshold used to determine whether two detections are duplicates
        net_h           : The height of the input image to the model, higher value results in better accuracy
        net_w           : The width of the input image to the model
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]

    for i in range(generator.size()):
        print(i, end='\r')
        raw_image = [generator.load_image(i)]

        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            try:
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            except IndexError:
                pass
    # compute mAP by comparing all detections and all annotations
    average_precisions = {}
    for label in range(generator.num_classes()):
        print()
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            print(i, end='\r')
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
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

    return average_precisions


# noinspection PyTypeChecker
def evaluate_acc(model,
                 generator,
                 iou_threshold=0.5,
                 obj_thresh=0.5,
                 nms_thresh=0.45,
                 net_h=416,
                 net_w=416,
                 save_path=None):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        obj_thresh      : The threshold used to distinguish between object and non-object
        nms_thresh      : The threshold used to determine whether two detections are duplicates
        net_h           : The height of the input image to the model, higher value results in better accuracy
        net_w           : The width of the input image to the model
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]

    for i in range(generator.size()):
        print(i, end='\r')
        raw_image = [generator.load_image(i)]

        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            try:
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            except IndexError:
                pass
    # compute mAP by comparing all detections and all annotations
    average_precisions = {}
    for label in range(generator.num_classes()):
        print()
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            print(i, end='\r')
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
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

    return average_precisions


def normalize(image):
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image = np.subtract(image.astype('float32'), MEAN_RGB)
    image = np.divide(image, STDDEV_RGB)
    return image  # effnet use this instead of image/255.


def draw_boxesv3(image, boxes, labels, obj_thresh):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 0, 255), (255, 255, 0), (0, 255, 255),
             (0, 0, 0), (255, 255, 255),
             ]

    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                # print(labels[i] + ': ' + str(box.classes[i] * 100) + '%')

        if label >= 0:
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), color[box.get_label() % 6], 1)
            cv2.putText(image,
                        label_str + ' ' + str(box.get_score()),
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        color[box.get_label() % 8], 1)

    return image


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(normalize(image[:, :, ::-1]), (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh, preprocess=True):
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    if preprocess:
        for i in range(nb_images):
            batch_input[i] = preprocess_input(images[i], net_h, net_w)

    # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            boxes += decode_netoutv3(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

    return batch_boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
