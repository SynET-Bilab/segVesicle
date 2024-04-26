import tensorflow as tf
import tensorflow.keras.backend as K


# with laplace smoothing
def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    mean_loss = (2. * intersection + smooth) / (union + smooth)
    return K.mean(mean_loss, axis=0)


# https://github.com/mkocabas/focal-loss-keras
def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))



def loss_use(y_true, y_pred):
    # balance the two losses, the focal loss should be controlled by lambda, a hyperparameter
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return focal_loss(y_true, y_pred, gamma=2., alpha=.25) - K.log(dice_coef(y_true, y_pred, smooth=1))
    #return focal_loss(y_true, y_pred, gamma=2., alpha=.25)
    #return 1- dice_coef(y_true, y_pred, smooth=1)