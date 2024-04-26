#!/usr/bin/env python

from tensorflow.keras.layers import LeakyReLU
from tomoSgmt.models.blocks.dropblock import DropBlock2D
from tomoSgmt.models.blocks.seblock import channel_spatial_squeeze_excite
from tomoSgmt.models.loss.loss_use import loss_use, dice_coef


def build_compiled_model(sidelen=128, neighbor_in=5, neighbor_out=1):

    '''
    U-Net with dropblock, activation=leakyrelu, loss=dice+focal_loss, metrics=[dice_coef]
    '''
    
    import tensorflow as tf
    import tensorflow.keras.layers as layers

    IMG_WIDTH = sidelen
    IMG_HEIGHT = sidelen
    IMG_CHANNELS = neighbor_in
    OUT_CHANNELS = neighbor_out

    inputs = layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = DropBlock2D(block_size=5, drop_rate=0.1)(conv1)
    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = DropBlock2D(block_size=5, drop_rate=0.1)(conv2)
    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = DropBlock2D(block_size=5, drop_rate=0.2)(conv3)
    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = DropBlock2D(block_size=5, drop_rate=0.2)(conv4)
    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = DropBlock2D(block_size=5, drop_rate=0.3)(conv5)
    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(up6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = DropBlock2D(block_size=5, drop_rate=0.2)(conv6)
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(up7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = DropBlock2D(block_size=5, drop_rate=0.2)(conv7)
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)

    up8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(up8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = DropBlock2D(block_size=5, drop_rate=0.1)(conv8)
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)

    up9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(up9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = DropBlock2D(block_size=5, drop_rate=0.1)(conv9)
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)

    outputs = layers.Conv2D(OUT_CHANNELS, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
    conv1 = DropBlock2D(block_size=5, drop_rate=0.1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = DropBlock2D(block_size=5, drop_rate=0.1)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = DropBlock2D(block_size=5, drop_rate=0.2)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = DropBlock2D(block_size=5, drop_rate=0.2)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = DropBlock2D(block_size=5, drop_rate=0.3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    # se block
    conv5 = channel_spatial_squeeze_excite(conv5)

    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(up6)
    conv6 = DropBlock2D(block_size=5, drop_rate=0.2)(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)

    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(up7)
    conv7 = DropBlock2D(block_size=5, drop_rate=0.2)(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)

    up8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(up8)
    conv8 = DropBlock2D(block_size=5, drop_rate=0.1)(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)

    up9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(up9)
    conv9 = DropBlock2D(block_size=5, drop_rate=0.1)(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)

    outputs = layers.Conv2D(OUT_CHANNELS, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_use, metrics=[dice_coef])

    return model


