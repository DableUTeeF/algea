import cv2
from yolo.utils import create_csv_training_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
from evaluate_util import evaluate, all_annotation_from_instance
import json


cfg = get_cfg()
cfg.merge_from_file("/home/palm/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("algea_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/palm/PycharmProjects/algea/detectron/output/model_0014999.pth"
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.DATASETS.TEST = ("balloon/val",)
predictor = DefaultPredictor(cfg)
config_path = '/home/palm/PycharmProjects/algea/yolo/algeaconfig.json'

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
        all_detection[1-classes[i]].append([*bboxes[i], scores[i]])
    # all_detection = np.array(all_detection, dtype='uint16')
    all_annotations.append(all_annotation)
    all_detections.append(all_detection)

average_precisions, total_instances = evaluate(all_detections, all_annotations, 2)
print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
    sum([a * b for a, b in zip(total_instances, average_precisions)]) / sum(total_instances)))
for label, average_precision in average_precisions.items():
    print(['ov', 'mif'][label] + ': {:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(sum(average_precisions) / sum(x > 0 for x in total_instances)))  # mAP: 0.5000
