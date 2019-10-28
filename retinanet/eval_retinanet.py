"""
05:
    mAP using the weighted average of precisions among classes: 0.7488
    mAP: 0.6741
08:
    mAP using the weighted average of precisions among classes: 0.6937
    mAP: 0.6220
"""
import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "retinanet"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .preprocessing.csv_generator import CSVGenerator
from .utils.eval import evaluate
import keras
from retinanet import losses
from retinanet import models
from retinanet.models.retinanet import retinanet_bbox
from retinanet.utils.config import parse_anchor_parameters
from retinanet.utils.model import freeze as freeze_model


def create_generator(annotations, classes, image_min_side, image_max_side):
    """ Create generators for evaluation.
    """
    validation_generator = CSVGenerator(
        annotations,
        classes,
        image_min_side=image_min_side,
        image_max_side=image_max_side,
    )
    return validation_generator


def create_models(backbone_retinanet, num_classes, freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model

    model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
    training_model = model
    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


if __name__ == '__main__':
    annotations = '/home/palm/PycharmProjects/algea/dataset/test_annotations'
    classes = '/home/palm/PycharmProjects/algea/dataset/classes'
    image_min_side = 1200
    image_max_side = 1600
    generator = create_generator(annotations, classes, image_min_side, image_max_side)

    model_path = '/home/palm/PycharmProjects/algea/snapshots/retina/r50-1/resnet50_csv_05.h5'

    backbone = models.backbone('resnet50')

    labels_to_names = {0: 'mif', 1: 'ov'}
    main_model, training_model, prediction_model = create_models(backbone.retinanet, len(labels_to_names))
    main_model.load_weights(model_path)
    model = prediction_model

    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
    )

    # print evaluation
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) == 0:
        print('No test instances found.')
        exit()

    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
        sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
    print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
