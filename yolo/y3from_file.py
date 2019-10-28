import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from yolo.y3frontend import *
import json
import time
from yolo.utils import draw_boxesv3, normalize, evaluate, evaluate_coco, get_yolo_boxes, parse_annotation, create_csv_training_instances
from PIL import Image
import numpy as np
from yolo.preprocessing import minmaxresize, Y3BatchGenerator
from keras.models import load_model

if __name__ == '__main__':

    config_path = 'algeaconfig.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        config['train']['train_csv'],
        config['valid']['valid_csv'],
        config['train']['classes_csv'],
    )

    infer_model = yolo3(
            fe='effnetb5',
            output_type='dw',
            nb_class=2
    )

    infer_model.load_weights('/home/palm/PycharmProjects/algea/snapshots/yolo/B5DW_algea_kaggle1/019_6.9314_4.8074.h5',
                             # by_name=True,
                             # skip_mismatch=True,
                             )

    path = "/media/palm/data/coco/images/val2017"
    pad = 1
    # for _ in range(1000):
    # if 1:

    valid_generator = Y3BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,
        max_box_per_image=max_box_per_image,
        batch_size=1,
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
    )
    average_precisions = evaluate(infer_model, valid_generator, net_h=608, net_w=608)
    #
    # for label, average_precision in average_precisions.items():
    #     print(labels[label] + ': {:.4f}'.format(average_precision))
    # print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    # raise SystemExit
    t = time.time()
    for idx, inst in enumerate(valid_ints):
    # for filename in os.listdir(path):
        # if 'txt' in filename:
        #     continue
        filename = inst['filename']
        # path = ''
        counter = 0
        p = os.path.join(filename)
        image = cv2.imread(p)
        x = time.time()
        # filename = '001dxxyile2uxkblr99uqo6fuhgprpccznlze0z0djhs9gkek2tsm8u5hsfzx62o.jpg'
        # filename = 'download.jpeg'
        image, w, h = minmaxresize(image, 608, 960)
        # image = cv2.resize(image, (416, 416))
        if pad:
            imsize = image.shape
            if imsize[0] > imsize[1]:
                tempim = np.zeros((imsize[0], imsize[0], 3), dtype='uint8')
                distant = (imsize[0] - imsize[1]) // 2
                tempim[:, distant:distant + imsize[1], :] = image
                image = tempim
                h = imsize[0]
                w = imsize[0]

            elif imsize[1] > imsize[0]:
                tempim = np.zeros((imsize[1], imsize[1], 3), dtype='uint8')
                distant = (imsize[1] - imsize[0]) // 2
                tempim[distant:distant + imsize[0], :, :] = image
                image = tempim
                h = imsize[1]
                w = imsize[1]

        image = np.expand_dims(image, 0)

        boxes = get_yolo_boxes(infer_model,
                               image,
                               608, 608,  # todo: change here too
                               config['model']['anchors'],
                               0.5,
                               0.5)[0]
        # infer_model.predict(image)
        # labels = ['badhelmet', 'badshoes', 'goodhelmet', 'goodshoes', 'person']
        # # draw bounding boxes on the image using labels
        image = draw_boxesv3(image[0], boxes, labels, 0.75)
        im = Image.fromarray(image.astype('uint8'))
        b, g, r = im.split()
        im = Image.merge("RGB", (r, g, b))
        print(time.time() - x)
        im.save('/home/palm/PycharmProjects/algea/dataset/yolo_testset/' + os.path.split(filename)[1])
    print('total time:', time.time() - t)
