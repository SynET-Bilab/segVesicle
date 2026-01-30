try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
except ImportError:
    pass

def dice_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  
    p1 = ones - y_pred  
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  
    Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
    return Ncl - T
