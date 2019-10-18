from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Activation, Flatten, Conv2D, BatchNormalization, \
    TimeDistributed, Lambda, DepthwiseConv2D, Multiply
import efficientnet
from keras import backend as K

from .RoiPoolingConv import RoiPoolingConv

EfficientNetConvInitializer = efficientnet.EfficientNetConvInitializer


def get_weight_path():
    return '/home/palm/.keras/models/efficientnet-b7_randaug.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


def td_se(input_filters, se_ratio, expand_ratio, block_name, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = TimeDistributed(Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True)))(x)
        x = TimeDistributed(Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True))(x)
        x = efficientnet.Swish(name=f'block{block_name}_SE_swish_1')(x)
        # Excite
        x = TimeDistributed(Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True))(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])
        return out

    return block


def td_mbconv(input_filters, output_filters,
              kernel_size, strides,
              expand_ratio, se_ratio,
              id_skip, drop_connect_rate,
              block_name,
              batch_norm_momentum=0.99,
              batch_norm_epsilon=1e-3,
              data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = TimeDistributed(Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False))(inputs)
            x = TimeDistributed(BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon))(x)
            x = efficientnet.Swish(name=f'block{block_name}_MB_swish_1')(x)
        else:
            x = inputs

        x = TimeDistributed(DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False))(x)
        x = TimeDistributed(BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon))(x)
        x = efficientnet.Swish(name=f'block{block_name}_MB_swish_2')(x)

        if has_se:
            x = td_se(input_filters, se_ratio, expand_ratio,
                      block_name, data_format)(x)

        # output phase

        x = TimeDistributed(Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False))(x)
        x = TimeDistributed(BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon))(x)

        return x

    return block


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    model = efficientnet.EfficientNetB7(input_tensor=img_input, include_top=False, pooling=None)
    return model.get_layer('block5_i_MB_swish_1').output


def classifier_layers(x):
    block_args_list = [
        efficientnet.BlockArgs(112, 192, kernel_size=5, strides=(2, 2), num_repeat=4, se_ratio=0.25, expand_ratio=6),
        efficientnet.BlockArgs(192, 320, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=6),
    ]
    round_filters = efficientnet.round_filters
    round_repeats = efficientnet.round_repeats

    width_coefficient = 2.0
    depth_coefficient = 3.1
    drop_connect_rate_per_block = 0.0
    depth_divisor = 8
    min_depth = None
    batch_norm_momentum = 0.99
    batch_norm_epsilon = 1e-3
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0
        block_idx += 6
        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor,
                                                  min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = td_mbconv(block_args.input_filters, block_args.output_filters,
                      block_args.kernel_size, block_args.strides,
                      block_args.expand_ratio, block_args.se_ratio,
                      block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                      str(block_idx) + 'new' + '_i',
                      batch_norm_momentum, batch_norm_epsilon, )(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = td_mbconv(block_args.input_filters, block_args.output_filters,
                          block_args.kernel_size, block_args.strides,
                          block_args.expand_ratio, block_args.se_ratio,
                          block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                          str(block_idx) + 'new' + '_' + str(_),
                          batch_norm_momentum, batch_norm_epsilon, )(x)
    return x


def rpn(base_layers, num_anchors):

    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
