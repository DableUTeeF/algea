import tensorflow as tf
from keras import layers, models
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from efficientnet import EfficientNetB3, EfficientNetB0, MBConvBlock
from keras.layers import Layer


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, writer=(None, None),
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale
        self.writer, self.layername = writer
        self.counter = 0
        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                              0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                       1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                       1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1],
                                  axis=4)  # the smaller the box, the bigger the scale

        xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * wh_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * wh_scale * self.xywh_scale
        conf_delta = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (
                1 - object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * tf.expand_dims(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class),
            4) * self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        # loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
        # loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
        #                        tf.reduce_sum(loss_wh),
        #                        tf.reduce_sum(loss_conf),
        #                        tf.reduce_sum(loss_class)], message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def _conv_block(inp, convs, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = layers.ZeroPadding2D(((1, 0), (1, 0)))(
                x)  # unlike tensorflow darknet prefer left and top paddings
        x = layers.Conv2D(conv['filter'],
                          conv['kernel'],
                          strides=conv['stride'],
                          padding='valid' if conv['stride'] > 1 else 'same',
                          # unlike tensorflow darknet prefer left and top paddings
                          name='conv_' + str(conv['layer_idx']),
                          use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']:
            x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']:
            x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return layers.add([skip_connection, x]) if do_skip else x


def _mb_conv_block(inp, convs, block_name):
    x = inp
    count = 0

    for conv in convs:
        count += 1
        x = MBConvBlock(conv['input_filters'],
                        conv['output_filters'],
                        conv['kernel'],
                        conv['stride'],
                        1,
                        0.25,
                        False,
                        0,
                        f'{block_name}_{count}_',
                        )(x)

    return x


def _dw_conv_block(inp, convs, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = layers.ZeroPadding2D(((1, 0), (1, 0)))(
                x)  # unlike tensorflow darknet prefer left and top paddings
        x = layers.SeparableConv2D(conv['filter'],
                                   conv['kernel'],
                                   strides=conv['stride'],
                                   padding='valid' if conv['stride'] > 1 else 'same',
                                   # unlike tensorflow darknet prefer left and top paddings
                                   name='conv_' + str(conv['layer_idx']),
                                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']:
            x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']:
            x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return layers.add([skip_connection, x]) if do_skip else x


def d53(input_image):
    # Layer  0 => 4
    x = _conv_block(input_image,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                     {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                     {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                     {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

    skip_36 = 'add_11'

    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

    skip_61 = 'add_19'

    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

    # Layer 75 => 79
    x = _conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                    do_skip=False)

    infer_model = models.Model(input_image, x)

    return infer_model, ['leaky_79', skip_61, skip_36]


def get_fe(fe, input_image):
    if fe == 'effnetb0':
        return EfficientNetB0(input_tensor=input_image, include_top=False), ['swish_last', 'block5_i_MB_swish_1',
                                                                             'block3_i_MB_swish_1']
    elif fe == 'effnetb3':
        return EfficientNetB3(input_tensor=input_image, include_top=False), ['swish_last', 'block5_i_MB_swish_1',
                                                                             'block3_i_MB_swish_1']
    elif fe == 'd53':
        return d53(input_image)
    elif fe == 'mnetv2':
        mnet = MobileNetV2(input_tensor=input_image, weights='imagenet')
        return mnet, ['out_relu', 'block_13_expand_relu', 'block_6_expand_relu']
    elif fe == 'mnet':
        mnet = MobileNet(input_tensor=input_image, weights='imagenet')
        return mnet, ['conv_pw_13_relu', 'conv_pw_11_relu', 'conv_pw_5_relu']
    elif fe == 'r50':
        r50 = ResNet50(input_tensor=input_image, weights='imagenet')
        return r50, ['activation_49', 'activation_40', 'activation_22']
    raise ValueError('Pls put the correct fe')


def common_pred_block_args(block_id):
    if block_id == 1:
        args = [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80}]
        return args
    elif block_id == 2:
        args = [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92}]
        return args
    elif block_id == 3:
        args = [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 99},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 100},
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 101},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 102},
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 103},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 104}]
        return args
    raise ValueError('Pls put a right block id')


def get_pred(x, block_id, output_type, nb_class):
    if output_type == 'conv':
        block = _conv_block
        args = common_pred_block_args(block_id)
        args.append({'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                     'layer_idx': [-1, 81, 93, 105][block_id]})
        return block(x, args, do_skip=False), block
    elif output_type == 'dw':
        block = _dw_conv_block
        args = common_pred_block_args(block_id)
        args.append({'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                     'layer_idx': [-1, 81, 93, 105][block_id]})
        return block(x, args, do_skip=False), block
    elif output_type == 'mb':
        block = _mb_conv_block
        args = mb_pred_block_args(block_id)
        args.append({'input_filters': [-1, 1024, 512, 256][block_id], 'output_filters': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1})
        return block(x, args, f'pred_yolo_{block_id}'), block
    raise ValueError('output_type is wrong')


def mb_pred_block_args(block_id):
    if block_id == 1:
        args = [{'input_filters': 1280, 'output_filters': 1024, 'kernel': 3, 'stride': 1}]
        return args
    elif block_id == 2:
        args = [{'input_filters': 256, 'output_filters': 512, 'kernel': 3, 'stride': 1}]
        return args
    elif block_id == 3:
        args = [{'input_filters': 368, 'output_filters': 128, 'kernel': 1, 'stride': 1},
                {'input_filters': 128, 'output_filters': 256, 'kernel': 3, 'stride': 1},
                {'input_filters': 256, 'output_filters': 128, 'kernel': 1, 'stride': 1},
                {'input_filters': 128, 'output_filters': 256, 'kernel': 3, 'stride': 1},
                {'input_filters': 256, 'output_filters': 128, 'kernel': 1, 'stride': 1},
                {'input_filters': 128, 'output_filters': 256, 'kernel': 3, 'stride': 1}]
        return args
    raise ValueError('Pls put a right block id')


def common_x_block_args(block_id):
    if block_id == 1:
        args = [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}]
        return args, False
    elif block_id == 2:
        args = [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}]
        return args, False
    elif block_id == 3:
        args = [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}]
        return args, False
    raise ValueError('Pls put a right block id')


def mb_x_block_args(block_id):
    if block_id == 1:
        args = [{'input_filters': 1280, 'output_filters': 256, 'kernel': 1, 'stride': 1}]
        return args, f'up_{block_id}'
    elif block_id == 2:
        args = [{'input_filters': 928, 'output_filters': 256, 'kernel': 1, 'stride': 1},
                {'input_filters': 256, 'output_filters': 512, 'kernel': 3, 'stride': 1},
                {'input_filters': 512, 'output_filters': 256, 'kernel': 1, 'stride': 1},
                {'input_filters': 256, 'output_filters': 512, 'kernel': 3, 'stride': 1},
                {'input_filters': 512, 'output_filters': 256, 'kernel': 1, 'stride': 1}]
        return args, f'up_{block_id}'
    elif block_id == 3:
        args = [{'input_filters': 256, 'output_filters': 128, 'kernel': 1, 'stride': 1}]
        return args, f'up_{block_id}'
    raise ValueError('Pls put a right block id')


def get_x_args(output_type, block_id):
    if output_type == 'mb':
        return mb_x_block_args(block_id)
    else:
        return common_x_block_args(block_id)


# todo
# noinspection PyUnboundLocalVariable
def yolo3(fe,
          output_type,
          nb_class,
          anchors=None,
          max_box_per_image=None,
          max_grid=None,
          batch_size=None,
          warmup_batches=None,
          ignore_thresh=None,
          grid_scales=None,
          obj_scale=None,
          noobj_scale=None,
          xywh_scale=None,
          class_scale=None,
          writer=None,
          infer_only=False):
    if not anchors:
        infer_only = True
    input_image = layers.Input(shape=(None, None, 3))
    true_boxes = layers.Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = layers.Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))
    true_yolo_2 = layers.Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))
    true_yolo_3 = layers.Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))

    fe, ls = get_fe(fe, input_image)
    x = fe.get_layer(ls[0]).output

    pred_yolo_1, block = get_pred(x, 1, output_type, nb_class)
    if not infer_only:
        loss_yolo_1 = YoloLayer(anchors[12:],
                                [1 * num for num in max_grid],
                                batch_size,
                                warmup_batches,
                                ignore_thresh,
                                grid_scales[0],
                                obj_scale,
                                noobj_scale,
                                xywh_scale,
                                class_scale,
                                (writer, 'yolo_1'))([input_image, pred_yolo_1, true_yolo_1, true_boxes])
    skip_2 = fe.get_layer(ls[1]).output
    x = block(x, *get_x_args(output_type, 1))
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_2])

    x = block(x, *get_x_args(output_type, 2))
    pred_yolo_2, _ = get_pred(x, 2, output_type, nb_class)
    if not infer_only:
        loss_yolo_2 = YoloLayer(anchors[6:12],
                                [2 * num for num in max_grid],
                                batch_size,
                                warmup_batches,
                                ignore_thresh,
                                grid_scales[1],
                                obj_scale,
                                noobj_scale,
                                xywh_scale,
                                class_scale,
                                (writer, 'yolo_2'))([input_image, pred_yolo_2, true_yolo_2, true_boxes])
    skip_3 = fe.get_layer(ls[2]).output
    x = block(x, *get_x_args(output_type, 3))
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_3])

    pred_yolo_3, _ = get_pred(x, 3, output_type, nb_class)
    if not infer_only:
        loss_yolo_3 = YoloLayer(anchors[:6],
                                [4 * num for num in max_grid],
                                batch_size,
                                warmup_batches,
                                ignore_thresh,
                                grid_scales[2],
                                obj_scale,
                                noobj_scale,
                                xywh_scale,
                                class_scale,
                                (writer, 'yolo_3'))([input_image, pred_yolo_3, true_yolo_3, true_boxes])
    if not infer_only:
        train_model = models.Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                                   [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = models.Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])
    if not infer_only:
        return [train_model, infer_model]
    else:
        return infer_model


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
