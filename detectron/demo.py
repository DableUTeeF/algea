import json
from detectron2.structures import BoxMode
import itertools
import cv2
from yolo.utils import create_csv_training_instances
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from boxutils import add_bbox


# write a function that loads the dataset into detectron2's standard format
def get_algea_dicts(img_dir):
    train_path = '/home/palm/PycharmProjects/algea/dataset/train_annotations'
    test_path = '/home/palm/PycharmProjects/algea/dataset/test_annotations'
    classes_path = '/home/palm/PycharmProjects/algea/dataset/classes'
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(train_path, test_path,
                                                                                      classes_path,
                                                                                      False)
    dataset_dicts = []
    for v in train_ints:
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        width, height = Image.open(filename).size

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        annos = v["object"]
        objs = []
        for anno in annos:
            px = (anno["xmin"], anno["xmax"])
            py = (anno["ymin"], anno["ymax"])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [anno["xmin"], anno["ymin"], anno["xmax"], anno["ymax"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


MetadataCatalog.get("algea").set(thing_classes=["algea"])
MetadataCatalog.get("algea_test").set(thing_classes=["algea"])
algea_metadata = MetadataCatalog.get("algea/train")

dataset_dicts = get_algea_dicts("/media/palm/data/MicroAlgae/16_8_62/images")

register_coco_instances("algea_train", {},
                        "/media/palm/data/MicroAlgae/16_8_62/annotations/train_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
register_coco_instances("algea_test", {},
                        "/media/palm/data/MicroAlgae/16_8_62/annotations/val_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
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
for d in valid_ints:
    im = cv2.imread(d["filename"])
    outputs = predictor(im)
    _field = outputs['instances']._fields
    bboxes = _field['pred_boxes']
    bboxes = bboxes.tensor
    bboxes = bboxes.cpu().numpy()
    classes = _field['pred_classes'].cpu().numpy()
    scores = _field['scores'].cpu().numpy()
    for i in range(len(bboxes)):
        color = [(0, 0, 255), (255, 0, 0)][classes[i] == int('mif' in d["filename"].lower())]
        if 'mif' in d["filename"].lower():
            if classes[i] == 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
        else:
            if classes[i] == 0:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

        im = add_bbox(im, bboxes[i].astype('uint16'), classes[i], ['ov', 'mif'], show_txt=True, color=color)
    for i in range(len(d['object'])):
        box = [d['object'][i]['xmin'], d['object'][i]['ymin'], d['object'][i]['xmax'], d['object'][i]['ymax']]
        im = add_bbox(im, box, 2, ['ov', 'mif', 'obj'], show_txt=False, color=(0, 200, 0))

    cv2.imwrite(os.path.join('/home/palm/PycharmProjects/algea/dataset/detectron_testset',
                             os.path.basename(d["filename"])),
                im
                )
