from detectron2.data.datasets import register_coco_instances
register_coco_instances("algea_train", {},
                        "/media/palm/data/MicroAlgae/16_8_62/annotations/train_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
register_coco_instances("algea_test", {},
                        "/media/palm/data/MicroAlgae/16_8_62/annotations/val_algea.json",
                        "/media/palm/data/MicroAlgae/16_8_62/images")
