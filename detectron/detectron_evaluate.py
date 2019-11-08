import cv2
from yolo.utils import create_csv_training_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
from evaluate_util import evaluate, all_annotation_from_instance
import json
import os
cfg = get_cfg()
cfg.merge_from_file("/home/palm/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("algea_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/palm/PycharmProjects/algea/detectron/output/model_0064999.pth"
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.DATASETS.TEST = ("balloon/val",)
predictor = DefaultPredictor(cfg)
config_path = '/home/palm/PycharmProjects/algea/yolo/algeaconfig.json'
exclude = ['MIF eggs-kato-40x (1).jpg',
           'MIF eggs-kato-40x (10).jpg',
           'MIF eggs-kato-40x (11).jpg',
           'MIF eggs-kato-40x (12).jpg',
           'MIF eggs-kato-40x (13).jpg',
           'MIF eggs-kato-40x (14).jpg',
           'MIF eggs-kato-40x (15).jpg',
           'MIF eggs-kato-40x (16).jpg',
           'MIF eggs-kato-40x (17).jpg',
           'MIF eggs-kato-40x (18).jpg',
           'MIF eggs-kato-40x (100).jpg',
           'MIF eggs-kato-40x (101).jpg',
           'MIF eggs-kato-40x (102).jpg',
           'MIF eggs-kato-40x (103).jpg',
           'MIF eggs-kato-40x (104).jpg',
           'MIF eggs-kato-40x (105).jpg',
           'MIF eggs-kato-40x (106).jpg',
           'MIF eggs-kato-40x (107).jpg',
           'MIF eggs-kato-40x (108).jpg',
           'MIF eggs-kato-40x (109).jpg',
           'MIF eggs-kato-40x (110).jpg',
           'MIF eggs-kato-40x (111).jpg',
           'MIF eggs-kato-40x (112).jpg',
           'MIF eggs-kato-40x (113).jpg',
           'MIF eggs-kato-40x (114).jpg',
           'MIF eggs-kato-40x (115).jpg',
           'MIF eggs-kato-40x (116).jpg',
           'MIF eggs-kato-40x (117).jpg',
           'MIF eggs-kato-40x (118).jpg',
           'MIF eggs-kato-40x (119).jpg',
           'MIF eggs-kato-40x (121).jpg',
           'MIF eggs-kato-40x (122).jpg',
           'MIF eggs-kato-40x (124).jpg',
           'MIF eggs-kato-40x (126).jpg',
           'MIF eggs-kato-40x (128).jpg',
           'MIF eggs-kato-40x (130).jpg',
           'MIF eggs-kato-40x (132).jpg',
           'MIF eggs-kato-40x (133).jpg',
           'MIF eggs-kato-40x (135).jpg',
           'MIF eggs-kato-40x (137).jpg',
           'MIF eggs-kato-40x (138).jpg',
           'MIF eggs-kato-40x (140).jpg',
           'MIF eggs-kato-40x (141).jpg',
           'MIF eggs-kato-40x (142).jpg',
           'MIF eggs-kato-40x (143).jpg',
           'MIF eggs-kato-40x (144).jpg',
           'MIF eggs-kato-40x (145).jpg',
           'MIF eggs-kato-40x (146).jpg',
           'MIF eggs-kato-40x (147).jpg',
           'MIF eggs-kato-40x (148).jpg',
           'MIF eggs-kato-40x (149).jpg',
           'MIF eggs-kato-40x (150).jpg',
           'MIF eggs-kato-40x (151).jpg',
           'MIF eggs-kato-40x (152).jpg',
           'MIF eggs-kato-40x (153).jpg',
           'MIF eggs-kato-40x (154).jpg',
           'MIF eggs-kato-40x (155).jpg',
           'MIF eggs-kato-40x (156).jpg',
           'MIF eggs-kato-40x (157).jpg',
           'MIF eggs-kato-40x (158).jpg',
           'MIF eggs-kato-40x (159).jpg',
           'MIF eggs-kato-40x (160).jpg',
           'MIF eggs-kato-40x (161).jpg',
           'MIF eggs-kato-40x (162).jpg',
           'MIF eggs-kato-40x (163).jpg',
           'MIF eggs-kato-40x (164).jpg',
           'MIF eggs-kato-40x (165).jpg',
           'MIF eggs-kato-40x (166).jpg',
           'MIF eggs-kato-40x (167).jpg',
           'MIF eggs-kato-40x (168).jpg',
           'MIF eggs-kato-40x (169).jpg',
           'MIF eggs-kato-40x (170).jpg',
           'MIF eggs-kato-40x (171).jpg',
           'MIF eggs-kato-40x (172).jpg',
           'MIF eggs-kato-40x (173).jpg',
           'MIF eggs-kato-40x (174).jpg',
           'MIF eggs-kato-40x (175).jpg',
           'MIF eggs-kato-40x (176).jpg',
           'MIF eggs-kato-40x (177).jpg',
           'MIF eggs-kato-40x (178).jpg',
           'MIF eggs-kato-40x (179).jpg',
           'MIF eggs-kato-40x (180).jpg', ]
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())
train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
    config['train']['train_csv'],
    config['valid']['valid_csv'],
    config['train']['classes_csv'],
)
all_detections = []
all_annotations = []
for instance in valid_ints:
    if os.path.basename(instance["filename"]) in exclude:
        print(os.path.basename(instance["filename"]))
        continue
    all_annotation = all_annotation_from_instance(instance)
    im = cv2.imread(instance["filename"])
    outputs = predictor(im)
    _field = outputs['instances']._fields
    bboxes = _field['pred_boxes']
    bboxes = bboxes.tensor
    bboxes = bboxes.cpu().numpy()
    classes = _field['pred_classes'].cpu().numpy()
    scores = _field['scores'].cpu().numpy()
    all_detection = [[], []]

    for i in range(len(bboxes)):
        all_detection[classes[i]].append([*bboxes[i], scores[i]])
    # all_detection = np.array(all_detection, dtype='uint16')
    all_annotations.append(all_annotation)
    all_detections.append(all_detection)

average_precisions, total_instances = evaluate(all_detections, all_annotations, 2)
print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
    sum([a * b for a, b in zip(total_instances, average_precisions)]) / sum(total_instances)))
for label, average_precision in average_precisions.items():
    print(['ov', 'mif'][label] + ': {:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(sum(average_precisions) / sum(x for x in total_instances)))  # mAP: 0.5000
