import keras
from retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.utils.visualization import draw_box, draw_caption
from PIL import Image
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import tensorflow as tf
from retinanet import losses
from retinanet import models
from retinanet.models.retinanet import retinanet_bbox
from retinanet.utils.config import parse_anchor_parameters
from retinanet.utils.model import freeze as freeze_model
from boxutils import add_bbox


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


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
    smin, smax = 1200, 1600

    keras.backend.tensorflow_backend.set_session(get_session())

    model_path = '/home/palm/PycharmProjects/algea/snapshots/retina/r50-1/resnet50_csv_05.h5'

    backbone = models.backbone('resnet50')

    labels_to_names = {0: 'mif', 1: 'ov'}
    main_model, training_model, prediction_model = create_models(backbone.retinanet, len(labels_to_names))
    main_model.load_weights(model_path)
    model = prediction_model
    path = '/media/palm/data/MicroAlgae/16_8_62/images'
    pad = 0
    ls = [s.split(',')[0] for s in open('/home/palm/PycharmProjects/algea/dataset/test_annotations').read().split('\n')]
    found = []
    # while True:
    # p = os.path.join(path, np.random.choice(os.listdir(path)))
    prd = len(os.listdir('/home/palm/PycharmProjects/algea/dataset/retina_testset'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd)))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd), 'correct'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd), 'wrong'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd), 'not_found'))
    os.mkdir(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd), 'mixed'))
    for pth in os.listdir(path):
        p = os.path.join(path, pth)
        if p not in ls:
            continue
        found.append(p)
        image = read_image_bgr(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.concatenate((np.expand_dims(image, 2), np.expand_dims(image, 2), np.expand_dims(image, 2)), 2)
        # copy raw image for license plate ocr
        raw_im = image.copy()

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=smin, max_side=smax)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        ptime = time.time() - start

        # correct for image scale
        boxes /= scale

        # visualize detections
        wrong_labels = []
        right_label = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            # color = label_color(label)
            if (label == 0 and 'mif' in pth.lower()) or (label == 1 and 'ov' in pth.lower()):
                right_label.append(label)
            else:
                wrong_labels.append(label)

            b = box.astype(int)
            # draw_box(draw, b, color=[(255, 0, 0), (0, 255, 0)][label])
            draw = add_bbox(draw, b, label, labels_to_names, score)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, b, caption)

        if len(wrong_labels) == 0 and len(right_label) > 0:
            flag = 'correct'
        elif len(wrong_labels) > 0 and len(right_label) > 0:
            flag = 'mixed'
        elif len(wrong_labels) == 0 and len(right_label) == 0:
            flag = 'not_found'
        else:
            flag = 'wrong'
        img = Image.fromarray(draw)
        img.save(os.path.join('/home/palm/PycharmProjects/algea/dataset/retina_testset', str(prd), flag, pth))
        print("processing time:", ptime, "nboxes:", boxes.shape)

    for l in ls:
        if l not in found:
            print(l)
