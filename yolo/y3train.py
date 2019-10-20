import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    __package__ = "yolo"

from datetime import datetime
import numpy as np
import json
from yolo.y3frontend import *
from yolo.preprocessing import Y3BatchGenerator
from yolo.utils import normalize, evaluate, makedirs, parse_annotation, create_csv_training_instances
from yolo.callbacks import CustomModelCheckpoint, CustomTensorBoard
import sys
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
# EarlyStopping = tf.keras.callbacks.EarlyStopping
# ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau


def create_callbacks(saved_weights_name, model_to_save, tensorboard_logs, config):
    makedirs(tensorboard_logs)

    def sc(epoch):
        if epoch < 10:
            return 1e-3
        elif epoch < 20:
            return 1e-4
        else:
            return 1e-5
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    makedirs(os.path.join(*os.path.split(saved_weights_name)[:-1]))
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='val_loss',
        verbose=1,
        # save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        patience=3,
        verbose=1,
        mode='min',
        min_lr=1e-6
    )
    now = datetime.now()
    dt = f'{now.year}-{now.month}-{now.day}_{now.hour}:{now.minute}:{now.second}'
    logdir = tensorboard_logs + os.path.split(saved_weights_name)[-2] + '_' + dt

    tensorboard = CustomTensorBoard(
        config,
        log_dir=logdir,
        write_graph=False,
        write_images=True,
    )
    return [checkpoint, early_stop, reduce_on_plateau, tensorboard]


def create_model(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid, batch_size,
        warmup_batches,
        ignore_thresh,
        multi_gpu,
        saved_weights_name,
        lr,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale,
        opt='rmsprop',
):
    template_model, infer_model = yolo3(
            fe='effnetb3',
            output_type='dw',
            nb_class=nb_class,
            anchors=anchors,
            max_box_per_image=max_box_per_image,
            max_grid=max_grid,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            ignore_thresh=ignore_thresh,
            grid_scales=grid_scales,
            obj_scale=obj_scale,
            noobj_scale=noobj_scale,
            xywh_scale=xywh_scale,
            class_scale=class_scale,
            # writer=writer
        )
    template_model.summary()
    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    # else:
    #     template_model.load_weights('backend.h5', by_name=True)

    train_model = template_model

    if opt == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=lr, clipnorm=0.001)
    elif opt == 'adam' or opt == 'Adam':
        optimizer = optimizers.Adam(lr=lr, clipnorm=0.001)
    else:
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, clipnorm=0.00)

    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


if __name__ == '__main__':

    config_path = 'yolo/algeaconfig.json'

    with open(config_path) as config_buffer:
        config_string = config_buffer.read()
        config = json.loads(config_string)

    ###############################
    #   Parse the annotations
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        config['train']['train_csv'],
        config['valid']['valid_csv'],
        config['train']['classes_csv'],
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators
    ###############################
    train_generator = Y3BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = Y3BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Create the model
    ###############################
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
        opt=config['train']['opt'],
    )
    callbacks = create_callbacks(config['train']['saved_weights_name'], infer_model, config['train']['tensorboard_dir'], config_string)
    ###############################
    #   Kick off the training
    ###############################
    print(f'\033[{np.random.randint(31, 37)}m')
    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=2,
        max_queue_size=8,
        validation_data=valid_generator,
        use_multiprocessing=False,
    )

    # make a GPU version of infer_model for evaluation
    infer_model.save_weights(config['train']['saved_weights_name']+'temp')
    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
