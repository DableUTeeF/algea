from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
import time
register_coco_instances("algea_train", {},
                        "/home/palm/PycharmProjects/algea/dataset/train_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
register_coco_instances("algea_test", {},
                        "/home/palm/PycharmProjects/algea/dataset/val_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
cfg = get_cfg()
cfg.merge_from_file("/home/palm/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("algea_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/palm/detectron2/pkls/R101-RetinaNet.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
time.sleep(600)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
