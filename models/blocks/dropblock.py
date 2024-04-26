'''
This file is from
'''

import tensorflow as tf

class DropBlock2D(tf.keras.layers.Layer):

    def __init__(self, drop_rate=0.2, block_size=3, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.rate = drop_rate
        self.block_size = block_size

    def call(self, inputs, training=None):
        if training:
            #batch size
            b = tf.shape(inputs)[0]
            
            random_tensor = tf.random.uniform(shape=[b, self.m_h, self.m_w, self.c]) + self.bernoulli_rate
            binary_tensor = tf.floor(random_tensor)
            binary_tensor = tf.pad(binary_tensor, [[0,0],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [self.block_size // 2, self.block_size // 2],
                                                   [0, 0]])
            binary_tensor = tf.nn.max_pool(binary_tensor,
                                           [1, self.block_size, self.block_size, 1],
                                           [1, 1, 1, 1],
                                           'SAME')
            binary_tensor = 1 - binary_tensor
            inputs = tf.math.divide(inputs, (1 - self.rate)) * binary_tensor
        return inputs
    
    def get_config(self):
        config = super(DropBlock2D, self).get_config()
        return config

    def build(self, input_shape):
        #feature map size (height, weight, channel)
        self.b, self.h, self.w, self.c = input_shape.as_list()
        #mask h, w
        self.m_h = self.h - (self.block_size // 2) * 2
        self.m_w = self.w - (self.block_size // 2) * 2
        self.bernoulli_rate = (self.rate * self.h * self.w) / (self.m_h * self.m_w * self.block_size**2)

# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer
# from scipy.stats import bernoulli
# import numpy as np
# import copy

# class DropBlock2D(Layer):
#     """
#     Regularization Technique for Convolutional Layers.

#     Pseudocode:
#     1: Input:output activations of a layer (A), block_size, γ, mode
#     2: if mode == Inference then
#     3: return A
#     4: end if
#     5: Randomly sample mask M: Mi,j ∼ Bernoulli(γ)
#     6: For each zero position Mi,j , create a spatial square mask with the center being Mi,j , the width,
#         height being block_size and set all the values of M in the square to be zero (see Figure 2).
#     7: Apply the mask: A = A × M
#     8: Normalize the features: A = A × count(M)/count_ones(M)

#     # Arguments
#         block_size: A Python integer. The size of the block to be dropped.
#         gamma: float between 0 and 1. controls how many activation units to drop.
#     # References
#         - [DropBlock: A regularization method for convolutional networks](
#            https://arxiv.org/pdf/1810.12890v1.pdf)
#     """
#     def __init__(self, block_size, keep_prob, **kwargs):
#         super(DropBlock2D, self).__init__(**kwargs)
#         self.block_size = block_size
#         self.keep_prob = keep_prob

#     def call(self, x, training=None):

#         # During inference, we do not Drop Blocks. (Similar to DropOut)
#         if training == None:
#             return x

#         # Calculate Gamma
#         feat_size = int(x.shape[-1])
#         gamma = ((1-self.keep_prob)/(self.block_size**2)) * ((feat_size**2) / ((feat_size-self.block_size+1)**2))

#         padding = self.block_size//2

#         # Randomly sample mask
#         sample = bernoulli.rvs(size=(feat_size-(padding*2), feat_size-(padding*2)),p=gamma)

#         # The above code creates a matrix of zeros and samples ones from the distribution
#         # We would like to flip all of these values
#         sample = 1-sample

#         # Pad the mask with ones
#         sample = np.pad(sample, pad_width=padding, mode='constant', constant_values=1)

#         # For each 0, create spatial square mask of shape (block_size x block_size)
#         mask = copy.copy(sample)
#         for i in range(feat_size):
#             for j in range(feat_size):
#                 if sample[i, j]==0:
#                     mask[i-padding : i+padding+1, j-padding : j+padding+1] = 0

#         mask = mask.reshape((1, feat_size, feat_size))

#         # Apply the mask
#         x = x * np.repeat(mask, x.shape[1], 0)

#         # Normalize the features
#         count = np.prod(mask.shape)
#         count_ones = np.count_nonzero(mask == 1)
#         x = x * count / count_ones

#         return x

#     def get_config(self):
#         config = {'block_size': self.block_size,
#                   'gamma': self.gamma,
#                   'seed': self.seed}
#         base_config = super(DropBlock2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def compute_output_shape(self, input_shape):
#         return input_shape


# from tensorflow import keras
# import tensorflow.keras.backend as K
# import tensorflow.compat.v1 as tf

# def _bernoulli(shape, mean):
#     return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

# class DropBlock2D(tf.keras.layers.Layer):
#     def __init__(self, keep_prob, block_size, scale=True, **kwargs):
#         super(DropBlock2D, self).__init__(**kwargs)
#         self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
#         self.block_size = int(block_size)
#         self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def build(self, input_shape):
#         assert len(input_shape) == 4
#         _, self.h, self.w, self.channel = input_shape.as_list()
#         # pad the mask
#         p1 = (self.block_size - 1) // 2
#         p0 = (self.block_size - 1) - p1
#         self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
#         self.set_keep_prob()
#         super(DropBlock2D, self).build(input_shape)

#     def call(self, inputs, training=None, **kwargs):
#         def drop():
#             mask = self._create_mask(tf.shape(inputs))
#             output = inputs * mask
#             output = tf.cond(self.scale,
#                              true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
#                              false_fn=lambda: output)
#             return output

#         if training is None:
#             training = K.learning_phase()
#         output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
#                          true_fn=lambda: inputs,
#                          false_fn=drop)
#         return output

#     def set_keep_prob(self, keep_prob=None):
#         """This method only supports Eager Execution"""
#         if keep_prob is not None:
#             self.keep_prob = keep_prob
#         w, h = tf.to_float(self.w), tf.to_float(self.h)
#         self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
#                      ((w - self.block_size + 1) * (h - self.block_size + 1))

#     def _create_mask(self, input_shape):
#         sampling_mask_shape = tf.stack([input_shape[0],
#                                        self.h - self.block_size + 1,
#                                        self.w - self.block_size + 1,
#                                        self.channel])
#         mask = _bernoulli(sampling_mask_shape, self.gamma)
#         mask = tf.pad(mask, self.padding)
#         mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
#         mask = 1 - mask
#         return mask


# class DropBlock2D(keras.layers.Layer):
#     """See: https://arxiv.org/pdf/1810.12890.pdf"""

#     def __init__(self,
#                  block_size,
#                  keep_prob,
#                  sync_channels=False,
#                  data_format=None,
#                  **kwargs):
#         """Initialize the layer.
#         :param block_size: Size for each mask block.
#         :param keep_prob: Probability of keeping the original feature.
#         :param sync_channels: Whether to use the same dropout for all channels.
#         :param data_format: 'channels_first' or 'channels_last' (default).
#         :param kwargs: Arguments for parent class.
#         """
#         super(DropBlock2D, self).__init__(**kwargs)
#         self.block_size = block_size
#         self.keep_prob = keep_prob
#         self.sync_channels = sync_channels
#         self.data_format = self.normalize_data_format(data_format)
#         self.input_spec = keras.layers.InputSpec(ndim=4)
#         self.supports_masking = True

#     def normalize_data_format(self, value):
#         if value is None:
#             value = K.image_data_format()
#         data_format = value.lower()
#         if data_format not in {'channels_first', 'channels_last'}:
#             raise ValueError('The `data_format` argument must be one of '
#                             '"channels_first", "channels_last". Received: ' +
#                             str(value))
#         return data_format

#     def get_config(self):
#         config = {'block_size': self.block_size,
#                   'keep_prob': self.keep_prob,
#                   'sync_channels': self.sync_channels,
#                   'data_format': self.data_format}
#         base_config = super(DropBlock2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def compute_mask(self, inputs, mask=None):
#         return mask

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def _get_gamma(self, height, width):
#         """Get the number of activation units to drop"""
#         height, width = K.cast(height, K.floatx()), K.cast(width, K.floatx())
#         block_size = K.constant(self.block_size, dtype=K.floatx())
#         return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
#                (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

#     def _compute_valid_seed_region(self, height, width):
#         positions = K.concatenate([
#             K.expand_dims(K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1),
#             K.expand_dims(K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1),
#         ], axis=-1)
#         half_block_size = self.block_size // 2
#         valid_seed_region = K.switch(
#             K.all(
#                 K.stack(
#                     [
#                         positions[:, :, 0] >= half_block_size,
#                         positions[:, :, 1] >= half_block_size,
#                         positions[:, :, 0] < height - half_block_size,
#                         positions[:, :, 1] < width - half_block_size,
#                     ],
#                     axis=-1,
#                 ),
#                 axis=-1,
#             ),
#             K.ones((height, width)),
#             K.zeros((height, width)),
#         )
#         return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

#     def _compute_drop_mask(self, shape):
#         height, width = shape[1], shape[2]
#         mask = K.random_binomial(shape, p=self._get_gamma(height, width))
#         mask *= self._compute_valid_seed_region(height, width)
#         mask = keras.layers.MaxPool2D(
#             pool_size=(self.block_size, self.block_size),
#             padding='same',
#             strides=1,
#             data_format='channels_last',
#         )(mask)
#         return 1.0 - mask

#     def call(self, inputs, training=None):

#         def dropped_inputs():
#             outputs = inputs
#             if self.data_format == 'channels_first':
#                 outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
#             shape = K.shape(outputs)
#             if self.sync_channels:
#                 mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
#             else:
#                 mask = self._compute_drop_mask(shape)
#             outputs = outputs * mask *\
#                 (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
#             if self.data_format == 'channels_first':
#                 outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
#             return outputs

#         return K.in_train_phase(dropped_inputs, inputs, training=training)