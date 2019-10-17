import os
import cv2
import copy
import numpy as np
import json
from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET
from .utils import BoundBox, bbox_iou, apply_random_scale_and_crop, random_distort_image, random_flip, \
    correct_bounding_boxes
from PIL import Image
import albumentations
import tensorflow as tf
import keras
Sequence = keras.utils.Sequence


def parse_json(ann_dir, img_dir):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        js = json.load(open(os.path.join(ann_dir, ann)))

        img['filename'] = img_dir + js['file_name']
        image = Image.open(img['filename'])
        img['width'], img['height'] = image.size

        for object_ in js['arr_boxes']:
            obj = {'name': object_['class']}

            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1

            img['object'] += [obj]

            obj['xmin'] = int(round(float(object_['x'])))
            obj['ymin'] = int(round(float(object_['y'])))
            obj['xmax'] = int(round(float(object_['x'] + object_['width'])))
            obj['ymax'] = int(round(float(object_['y'] + object_['height'])))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None,
                 flipflop=True,
                 shoechanger=True,
                 zeropad=True
                 ):
        self.generator = None

        self.flipflop = flipflop
        self.shoechanger = shoechanger
        if self.flipflop or self.shoechanger:
            self.badshoes = []
            for im in os.listdir('imgs/more_badshoes'):
                self.badshoes.append(cv2.imread('imgs/more_badshoes/' + im))

        self.zeropad = zeropad

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i in
                        range(int(len(config['ANCHORS']) // 2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    # shear=(-5, 5), # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'],
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'],
                            4 + 1 + len(self.config['LABELS'])))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, self.config['IMAGE_H'], self.config['IMAGE_W'])

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (
                                float(self.config['IMAGE_W']) / self.config['GRID_W'])  # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (
                                float(self.config['IMAGE_H']) / self.config['GRID_H'])  # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:, :, ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[:, :, ::-1], obj['name'],
                                    (obj['xmin'] + 2, obj['ymin'] + 12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

            # print(' new batch created', idx)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image

        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)

        new_h = self.config['IMAGE_H']
        new_w = self.config['IMAGE_W']

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w,
                                          image_h)

        return im_sized, all_objs

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])
        changeable = []
        hashelmet = False

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            ### translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

            # resize the image to standard size

        if self.zeropad:
            imsize = image.shape
            if imsize[0] > imsize[1]:
                tempim = np.zeros((imsize[0], imsize[0], 3), dtype='uint8')
                distant = (imsize[0] - imsize[1]) // 2
                tempim[:, distant:distant + imsize[1], :] = image
                image = tempim
                h = imsize[0]
                w = imsize[0]
                for obj in all_objs:
                    obj['xmin'] += distant
                    obj['xmax'] += distant

            elif imsize[1] > imsize[0]:
                tempim = np.zeros((imsize[1], imsize[1], 3), dtype='uint8')
                distant = (imsize[1] - imsize[0]) // 2
                tempim[distant:distant + imsize[0], :, :] = image
                image = tempim
                h = imsize[1]
                w = imsize[1]
                for obj in all_objs:
                    obj['ymin'] += distant
                    obj['ymax'] += distant

        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:, :, ::-1]

        # fix object's position and size
        for idx, obj in enumerate(all_objs):
            if obj['name'] == 'goodshoes':
                changeable.append(idx)
            if obj['name'] == 'helmet':
                hashelmet = True

            for attr in ['xmin', 'xmax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        if self.flipflop and np.random.rand() > 0.85:
            im = copy.deepcopy(np.random.choice(self.badshoes))
            randsize = np.random.rand(2)
            ssize = self.config['IMAGE_H'] // 8
            im = cv2.resize(im, (ssize + int(ssize / 2 * randsize[0]), ssize + int(ssize / 2 * randsize[1]))).astype(
                'float32')
            # im = self.aug_pipe.augment_image(im)
            im = im.astype('uint8')
            loc = (np.random.rand(2) * self.config['IMAGE_H']).astype('uint8')
            image[loc[0]:loc[0] + im.shape[0], loc[1]:loc[1] + im.shape[1], :] = im
            all_objs.append({'xmin': loc[0],
                             'ymin': loc[1],
                             'xmax': loc[0] + im.shape[0],
                             'ymax': loc[1] + im.shape[1],
                             'name': 'badshoes'
                             })

        if self.shoechanger and hashelmet:
            im = copy.deepcopy(np.random.choice(self.badshoes))
            for idx in changeable:
                if np.random.rand() < 0.85:
                    continue
                obj = all_objs[idx]
                xsize = int(obj['xmax'] - obj['xmin'])
                ysize = int(obj['ymax'] - obj['ymin'])
                im = cv2.resize(im, (xsize, ysize))
                # im = self.aug_pipe.augment_image(im)
                image[obj['ymin']:obj['ymin'] + ysize, obj['xmin']:obj['xmin'] + xsize, :] = im

        return image, all_objs


class Y3BatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 anchors,
                 labels,
                 downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                 max_box_per_image=30,
                 batch_size=1,
                 min_net_size=608,
                 max_net_size=800,
                 shuffle=True,
                 jitter=True,
                 norm=None
                 ):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = (min_net_size // self.downsample) * self.downsample
        self.max_net_size = (max_net_size // self.downsample) * self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = max_net_size
        self.net_w = max_net_size

        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        net_h, net_w = self.net_h, self.net_w
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))  # input images
        gt_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                        # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                gt_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 3)
                    cv2.putText(img, obj['name'],
                                (obj['xmin'] + 2, obj['ymin'] + 12),
                                0, 1.2e-3 * img.shape[0],
                                (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        return [x_batch, gt_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(self.min_net_size / self.downsample,
                                                           self.max_net_size / self.downsample + 1)
            # print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image

        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w,
                                          image_h)

        return im_sized, all_objs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])


class BatchGenS1(BatchGenerator):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None,
                 flipflop=True,
                 shoechanger=True,
                 zeropad=True
                 ):
        self.generator = None

        self.flipflop = flipflop
        self.shoechanger = shoechanger
        if self.flipflop or self.shoechanger:
            self.badshoes = []
            for im in os.listdir('imgs/more_badshoes'):
                self.badshoes.append(cv2.imread('imgs/more_badshoes/' + im))

        self.zeropad = zeropad

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i in
                        range(int(len(config['ANCHORS']) // 2))]
        self.labels_to_names = {0: 'goodhelmet', 1: 'LP', 2: 'goodshoes', 3: 'badshoes', 4: 'badhelmet', 5: 'person'}
        self.names_to_labels = {'goodhelmet': 0, 'LP': 1, 'goodshoes': 2, 'badshoes': 3, 'badhelmet': 4, 'person': 5}

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        train_instance = self.images[idx]
        # augment input image and fix object's position and size
        augmented = self.aug_image(train_instance)
        x_batch = np.expand_dims(augmented['image'], 0)
        net_h, net_w = augmented['image'].shape[0:2]

        b_batch = np.zeros((1, 1, 1, 1, self.config['TRUE_BOX_BUFFER'],
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((1, net_h // 32, net_w // 32, self.config['BOX'],
                            4 + 1 + len(self.config['LABELS'])))  # desired network output

        # construct output from object's x, y, w, h
        true_box_index = 0

        for idx2, (xmin, ymin, xmax, ymax) in enumerate(augmented['bboxes']):
            if xmax > xmin and ymax > ymin and augmented['category_id'][idx2] in self.config['LABELS']:
                center_x = .5 * (xmin + xmax)
                center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                center_y = .5 * (ymin + ymax)
                center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    obj_indx = augmented['category_id'][idx2]

                    center_w = (xmax - xmin) / (
                            float(self.config['IMAGE_W']) / self.config['GRID_W'])  # unit: grid cell
                    center_h = (ymax - ymin) / (
                            float(self.config['IMAGE_H']) / self.config['GRID_H'])  # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[0, grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[0, grid_y, grid_x, best_anchor, 4] = 1.
                    y_batch[0, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                    # assign the true box to b_batch
                    b_batch[0, 0, 0, 0, true_box_index] = box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

        # assign input image to x_batch
        x_batch = self.norm(x_batch)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def aug_image(self, instance):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        image, new_w, new_h = minmaxresize(image, 352, 544)

        # determine the amount of scaling and cropping
        dw = image_w - (self.jitter * image_w)
        dh = image_h - (self.jitter * image_h)
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, new_w, new_h, 0, 0, 0, image_w,
                                          image_h)

        # alabumentation setup
        annotations = {'image': image, 'bboxes': [],
                       'category_id': []}
        for ann in all_objs:
            annotations['category_id'].append(self.names_to_labels[ann['name']])
            annotations['bboxes'].append(
                [ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
        aug = albumentations.Compose([albumentations.RGBShift(),
                                      albumentations.HorizontalFlip(),
                                      # albumentations.ShiftScaleRotate(scale_limit=1.),
                                      albumentations.CLAHE(),
                                      albumentations.RandomGamma(),
                                      ],
                                     bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})
        augmented = aug(**annotations)
        return augmented


class Y3BatchGeneratorS1(Sequence):
    def __init__(self,
                 instances,
                 anchors,
                 labels,
                 downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                 max_box_per_image=30,
                 batch_size=1,
                 min_net_size=320,
                 max_net_size=608,
                 shuffle=True,
                 jitter=True,
                 norm=None
                 ):
        self.instances = instances
        self.batch_size = 1
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = (min_net_size // self.downsample) * self.downsample
        self.max_net_size = (max_net_size // self.downsample) * self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.labels_to_names = {0: 'goodhelmet', 1: 'LP', 2: 'goodshoes', 3: 'badshoes', 4: 'badhelmet', 5: 'person'}
        self.names_to_labels = {'goodhelmet': 0, 'LP': 1, 'goodshoes': 2, 'badshoes': 3, 'badhelmet': 4, 'person': 5}

        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        # do the logic to fill in the inputs and the output
        train_instance = self.instances[idx]
        # augment input image and fix object's position and size
        # img, all_objs = self._aug_image(train_instance, net_h, net_w)
        augmented = self._aug_image(train_instance)

        net_h, net_w = augmented['image'].shape[0:2]
        x_batch = np.expand_dims(augmented['image'], 0)

        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample
        gt_batch = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((1, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((1, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((1, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((1, 1))
        dummy_yolo_2 = np.zeros((1, 1))
        dummy_yolo_3 = np.zeros((1, 1))

        instance_count = 0
        true_box_index = 0

        for idx2, (xmin, ymin, xmax, ymax) in enumerate(augmented['bboxes']):
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            # find the best anchor box for this object
            max_anchor = None
            max_index = -1
            max_iou = -1

            shifted_box = BoundBox(0,
                                   0,
                                   xmax - xmin,
                                   ymax - ymin)

            for i in range(len(self.anchors)):
                anchor = self.anchors[i]
                iou = bbox_iou(shifted_box, anchor)

                if max_iou < iou:
                    max_anchor = anchor
                    max_index = i
                    max_iou = iou

                    # determine the yolo to be responsible for this bounding box
            yolo = yolos[max_index // 3]
            grid_h, grid_w = yolo.shape[1:3]

            # determine the position of the bounding box on the grid
            center_x = .5 * (xmin + xmax)
            center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
            center_y = .5 * (ymin + ymax)
            center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

            # determine the sizes of the bounding box
            w = np.log((xmax - xmin) / float(max_anchor.xmax))  # t_w
            h = np.log((ymax - ymin) / float(max_anchor.ymax))  # t_h

            box = [center_x, center_y, w, h]

            # determine the index of the label
            obj_indx = augmented['category_id'][idx2]

            # determine the location of the cell responsible for this object
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            yolo[0, grid_y, grid_x, max_index % 3] = 0
            yolo[0, grid_y, grid_x, max_index % 3, 0:4] = box
            yolo[0, grid_y, grid_x, max_index % 3, 4] = 1.
            yolo[0, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

            # assign the true box to t_batch
            true_box = [center_x, center_y, xmax - xmin, ymax - ymin]
            gt_batch[0, 0, 0, 0, true_box_index] = true_box

            true_box_index += 1
            true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            x_batch = self.norm(x_batch)

        return [x_batch, gt_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _aug_image(self, instance):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        image, new_w, new_h = minmaxresize(image, self.min_net_size, self.max_net_size)

        # determine the amount of scaling and cropping
        dw = image_w - (self.jitter * image_w)
        dh = image_h - (self.jitter * image_h)
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, new_w, new_h, 0, 0, 0, image_w,
                                          image_h)

        # alabumentation setup
        annotations = {'image': image, 'bboxes': [],
                       'category_id': []}
        for ann in all_objs:
            annotations['category_id'].append(self.names_to_labels[ann['name']])
            annotations['bboxes'].append(
                [ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
        aug = albumentations.Compose([albumentations.HorizontalFlip(),
                                      # albumentations.RGBShift(),
                                      # albumentations.ShiftScaleRotate(scale_limit=1.),
                                      # albumentations.CLAHE(),
                                      # albumentations.RandomGamma(),
                                      ],
                                     bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})
        augmented = aug(**annotations)
        return augmented

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])


class Y3Gen2(Y3BatchGenerator):
    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image
        image_h, image_w, _ = image.shape

        image, new_w, new_h = minmaxresize(image, self.min_net_size, self.min_net_size)

        # determine the amount of scaling and cropping
        dw = image_w - (self.jitter * image_w)
        dh = image_h - (self.jitter * image_h)
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, new_w, new_h, 0, 0, 0, image_w,
                                          image_h)
        labels = {}
        for i, l in enumerate(self.labels):
            labels[l] = i
        # alabumentation setup
        annotations = {'image': image, 'bboxes': [],
                       'category_id': []}
        for ann in all_objs:
            annotations['category_id'].append(labels[ann['name']])
            annotations['bboxes'].append(
                [ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
        aug = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.RGBShift(),
            albumentations.RandomCrop(min(new_h, new_w), min(new_h, new_w))
            # albumentations.ShiftScaleRotate(scale_limit=1.),
            # albumentations.CLAHE(),
            # albumentations.RandomGamma(),
        ],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})
        augmented = aug(**annotations)
        all_objs = []
        for i in range(len(augmented['bboxes'])):
            ann = augmented['bboxes'][i]
            xmin, ymin, xmax, ymax = ann[0], ann[1], ann[2], ann[3]
            name = self.labels[augmented['category_id'][i]]
            all_objs.append({'name': name, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})
        return augmented['image'], all_objs


def minmaxresize(image, minside, maxside):
    if maxside >= image.shape[0] >= minside and maxside >= image.shape[1] >= minside:
        h = image.shape[0] // 32 * 32
        w = image.shape[1] // 32 * 32
        image = cv2.resize(image, (w, h))
        return image, w, h

    if image.shape[1] > maxside:
        w = maxside
        h = minside if w / image.shape[1] * image.shape[0] < minside else w / image.shape[1] * image.shape[0] // 32 * 32
    elif image.shape[0] > maxside:
        h = maxside
        w = minside if h / image.shape[0] * image.shape[1] < minside else h / image.shape[0] * image.shape[1] // 32 * 32
    elif image.shape[1] < minside:
        w = minside
        h = maxside if w / image.shape[1] * image.shape[0] > maxside else w / image.shape[1] * image.shape[0] // 32 * 32
    elif image.shape[0] < minside:
        h = minside
        w = maxside if h / image.shape[0] * image.shape[1] > maxside else h / image.shape[0] * image.shape[1] // 32 * 32
    else:
        h = image.shape[0] // 32 * 32
        w = image.shape[1] // 32 * 32
    w, h = int(w), int(h)
    image = cv2.resize(image, (w, h))
    return image, w, h
