#!/usr/bin/env python

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    mean_loss = 0
    for i in range(y_pred.shape(-1)):
        intersection = K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
        union = K.sum(y_true[:,:,:,i], axis=[1,2,3]) + K.sum(y_pred[:,:,:,i], axis=[1,2,3])
    mean_loss += (2. * intersection + smooth) / (union + smooth)
    return K.mean(mean_loss, axis=0)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


# def loss_use(smooth=1, gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         mean_loss = 0
#         for i in range(y_pred.shape(-1)):
#             intersection = K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
#             union = K.sum(y_true[:,:,:,i], axis=[1,2,3]) + K.sum(y_pred[:,:,:,i], axis=[1,2,3])
#         mean_loss += (2. * intersection + smooth) / (union + smooth)
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())) + 1-K.mean(mean_loss, axis=0)
#     return focal_loss_fixed


def loss_use(y_true, y_pred):
    return focal_loss(gamma=2., alpha=.25) + 1-dice_coef(y_true, y_pred, smooth=1)


class DropBlock2D(keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = self.normalize_data_format(data_format)
        self.input_spec = keras.layers.InputSpec(ndim=4)
        self.supports_masking = True

    def normalize_data_format(self, value):
        if value is None:
            value = K.image_data_format()
        data_format = value.lower()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('The `data_format` argument must be one of '
                            '"channels_first", "channels_last". Received: ' +
                            str(value))
        return data_format

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, height, width):
        """Get the number of activation units to drop"""
        height, width = K.cast(height, K.floatx()), K.cast(width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((height, width)),
            K.zeros((height, width)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        height, width = shape[1], shape[2]
        mask = K.random_binomial(shape, p=self._get_gamma(height, width))
        mask *= self._compute_valid_seed_region(height, width)
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


# def build_compiled_model(sidelen=128,neighbor_in=5,neighbor_out=1):
#     #Build the model
#     IMG_WIDTH = sidelen
#     IMG_HEIGHT = sidelen
#     IMG_CHANNELS = neighbor_in
#     OUT_CHANNELS = neighbor_out
#     inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#     s = inputs
#     #Contraction path
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
#     c1 = DropBlock2D(block_size=5, keep_prob=0.9)(c1)
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = DropBlock2D(block_size=5, keep_prob=0.9)(c2)
#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = DropBlock2D(block_size=5, keep_prob=0.8)(c3)
#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = DropBlock2D(block_size=5, keep_prob=0.8)(c4)
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     c5 = DropBlock2D(block_size=5, keep_prob=0.7)(c5)
#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#     #Expansive path 
#     u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = tf.keras.layers.concatenate([u6, c4])
#     c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = DropBlock2D(block_size=5, keep_prob=0.8)(c6)
#     c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
#     u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = tf.keras.layers.concatenate([u7, c3])
#     c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = DropBlock2D(block_size=5, keep_prob=0.8)(c7)
#     c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
#     u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = tf.keras.layers.concatenate([u8, c2])
#     c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = DropBlock2D(block_size=5, keep_prob=0.9)(c8)
#     c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
#     u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
#     c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = DropBlock2D(block_size=5, keep_prob=0.9)(c9)
#     c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    
#     outputs = tf.keras.layers.Conv2D(OUT_CHANNELS, (1, 1), activation='sigmoid')(c9)
    
#     model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer='adam', loss=loss_use, metrics=[dice_coef])

#     return model


def build_compiled_model(sidelen=128, neighbor_in=5, neighbor_out=1):

    '''
    U-Net with dropblock, activation=leakyrelu, loss=dice+focal_loss, metrics=[dice_coef]
    '''
    
    import tensorflow as tf
    import tensorflow.keras.layers as layers
    from tensorflow.keras.layers import LeakyReLU

    IMG_WIDTH = sidelen
    IMG_HEIGHT = sidelen
    IMG_CHANNELS = neighbor_in
    OUT_CHANNELS = neighbor_out

    inputs = layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = DropBlock2D(block_size=5, keep_prob=0.9)(conv1)
    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = DropBlock2D(block_size=5, keep_prob=0.9)(conv2)
    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = DropBlock2D(block_size=5, keep_prob=0.8)(conv3)
    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = DropBlock2D(block_size=5, keep_prob=0.8)(conv4)
    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = DropBlock2D(block_size=5, keep_prob=0.7)(conv5)
    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(up6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = DropBlock2D(block_size=5, keep_prob=0.8)(conv6)
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(up7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = DropBlock2D(block_size=5, keep_prob=0.8)(conv7)
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)

    up8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(up8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = DropBlock2D(block_size=5, keep_prob=0.9)(conv8)
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)

    up9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(up9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = DropBlock2D(block_size=5, keep_prob=0.9)(conv9)
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)

    outputs = layers.Conv2D(OUT_CHANNELS, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_use, metrics=[dice_coef])

    return model


def build_compiled_model_se(sidelen=128, neighbor_in=5, neighbor_out=1):

    '''
    U-Net with dropblock and se block, activation=leakyrelu, loss=dice+focal_loss, metrics=[dice_coef]
    '''

    import tensorflow as tf
    import tensorflow.keras.layers as layers
    from tensorflow.keras.layers import LeakyReLU

    IMG_WIDTH = sidelen
    IMG_HEIGHT = sidelen
    IMG_CHANNELS = neighbor_in
    OUT_CHANNELS = neighbor_out

    inputs = layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = DropBlock2D(block_size=5, keep_prob=0.9)(conv1)
    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = DropBlock2D(block_size=5, keep_prob=0.9)(conv2)
    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = DropBlock2D(block_size=5, keep_prob=0.8)(conv3)
    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = DropBlock2D(block_size=5, keep_prob=0.8)(conv4)
    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = DropBlock2D(block_size=5, keep_prob=0.7)(conv5)
    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    # se block
    conv5 = channel_spatial_squeeze_excite(conv5)

    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(up6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = DropBlock2D(block_size=5, keep_prob=0.8)(conv6)
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(up7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = DropBlock2D(block_size=5, keep_prob=0.8)(conv7)
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)

    up8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(up8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = DropBlock2D(block_size=5, keep_prob=0.9)(conv8)
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)

    up9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(up9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = DropBlock2D(block_size=5, keep_prob=0.9)(conv9)
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)

    outputs = layers.Conv2D(OUT_CHANNELS, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_use, metrics=[dice_coef])

    return model


# git@github.com:titu1994/keras-squeeze-excite-network.git

def _tensor_shape(tensor):
    return getattr(tensor, 'shape')


def squeeze_excite_block(input_tensor, ratio=16):

    """ Create a channel-wise squeeze-excite block

    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters

    Returns: a Keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block

    Args:
        input_tensor: input Keras tensor

    Returns: a Keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block

    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters

    Returns: a Keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x