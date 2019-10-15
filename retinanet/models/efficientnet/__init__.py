from .efficientnet import *
from .custom_objects import *
from .config import *


import keras

from .. import retinanet
from .. import Backbone

allowed_backbones = {
    'efficientnetb0': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB0),
    'efficientnetb1': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB1),
    'efficientnetb2': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB2),
    'efficientnetb3': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB3),
    'efficientnetb4': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB4),
    'efficientnetb5': (('block3_i_MB_swish_1', 'block5_i_MB_swish_1', 'swish_last'), EfficientNetB5),
}


class EfficientNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return efficientnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        pass

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones.keys()))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_input(inputs)


def efficientnet_retinanet(num_classes, backbone='efficientnetb3', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a densenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a DenseNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    blocks, creator = allowed_backbones[backbone]
    model = creator(input_tensor=inputs, include_top=False, pooling=None, weights=None)

    # get last conv layer from the end of each dense block
    layer_outputs = [model.get_layer(name=block_name).output for block_name in blocks]

    # create the densenet backbone
    model = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=model.name)

    # invoke modifier if given
    if modifier:
        model = modifier(model)

    # create the full model
    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=model.outputs, **kwargs)

    return model
