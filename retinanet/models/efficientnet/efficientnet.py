# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
import math
from typing import List

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file, get_source_inputs

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess

from .config import BlockArgs, get_default_block_list
from .custom_objects import EfficientNetConvInitializer
from .custom_objects import EfficientNetDenseInitializer
from .custom_objects import Swish, DropConnect

__all__ = ['EfficientNet',
           'EfficientNetB0',
           'EfficientNetB1',
           'EfficientNetB2',
           'EfficientNetB3',
           'EfficientNetB4',
           'EfficientNetB5',
           'EfficientNetB6',
           'EfficientNetB7',
           'preprocess_input']

import keras


class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # return super.call, but set training
        return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


def preprocess_input(x, data_format=None):
    return _preprocess(x, data_format, mode='torch', backend=K)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, block_name, data_format=None):
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
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Swish(name=f'block{block_name}_SE_swish_1')(x)
        # Excite
        x = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(input_filters, output_filters,
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
            x = layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = BatchNormalization(True,
                                   axis=channel_axis,
                                   momentum=batch_norm_momentum,
                                   epsilon=batch_norm_epsilon)(x)
            x = Swish(name=f'block{block_name}_MB_swish_1')(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(True,
                               axis=channel_axis,
                               momentum=batch_norm_momentum,
                               epsilon=batch_norm_epsilon)(x)
        x = Swish(name=f'block{block_name}_MB_swish_2')(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        block_name, data_format)(x)

        # output phase

        x = layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(True,
                               axis=channel_axis,
                               momentum=batch_norm_momentum,
                               epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def EfficientNet(input_shape,
                 block_args_list: List[BlockArgs],
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=None,
                 **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_default_block_list()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=min_size,
                                      data_format=data_format,
                                      require_flatten=include_top,
                                      weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    x = layers.Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization(True,
                           axis=channel_axis,
                           momentum=batch_norm_momentum,
                           epsilon=batch_norm_epsilon)(x)
    x = Swish(name='swish_0')(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor,
                                                  min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        str(block_idx) + '_i',
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            str(block_idx) + '_' + str(_),
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

    # Head part
    x = layers.Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization(True,
                           axis=channel_axis,
                           momentum=batch_norm_momentum,
                           epsilon=batch_norm_epsilon)(x)
    x = Swish(name='swish_last')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    # Load weights
    if weights == 'imagenet':
        if default_size == 224:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b0.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b0_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 240:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b1.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b1_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 260:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b2.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b2_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 300:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b3.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b3_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 380:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b4.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b4_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        elif default_size == 456:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b5.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5.h5",
                    cache_subdir='models')
            else:
                weights_path = get_file(
                    'efficientnet-b5_notop.h5',
                    "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5_notop.h5",
                    cache_subdir='models')
            model.load_weights(weights_path)

        # elif default_size == 528:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b6.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6.h5",
        #             cache_subdir='models')
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b6_notop.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6_notop.h5",
        #             cache_subdir='models')
        #     model.load_weights(weights_path)
        #
        # elif default_size == 600:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b7.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7.h5",
        #             cache_subdir='models')
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b7_notop.h5',
        #             "https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7_notop.h5",
        #             cache_subdir='models')
        #     model.load_weights(weights_path)

        else:
            raise ValueError('ImageNet weights can only be loaded with EfficientNetB0-7')

    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B0.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=224)


def EfficientNetB1(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B1.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=240)


def EfficientNetB2(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B2.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=260)


def EfficientNetB3(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B3.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.2,
                        depth_coefficient=1.4,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=300)


def EfficientNetB4(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B4.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.4,
                        depth_coefficient=1.8,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=380)


def EfficientNetB5(input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B5.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.6,
                        depth_coefficient=2.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=456)


def EfficientNetB6(input_shape=None,
                   include_top=True,
                   weights=None,
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B6.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.8,
                        depth_coefficient=2.6,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=528)


def EfficientNetB7(input_shape=None,
                   include_top=True,
                   weights=None,
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B7.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=2.0,
                        depth_coefficient=3.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=600)


if __name__ == '__main__':
    import os
    from keras.models import load_model

    model = EfficientNetB0(include_top=True)
    model.summary()

    model.save("temp.h5")

    if os.path.exists('temp.h5'):
        model = load_model('temp.h5', compile=False)
        model.summary()

    else:
        raise FileNotFoundError("Keras model file not found !")

    if os.path.exists('temp.h5'):
        os.remove('temp.h5')
