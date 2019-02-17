import numpy as np
import tensorflow as tf

def conv2d_layer(x, filters, kernel_size, strides, without_padding=False, use_bias=False):
    if without_padding:
        padding = 'VALID'
    elif strides > 1:
        padding = 'VALID'
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    else:
        padding = 'SAME'
    return tf.layers.conv2d(
        inputs=x, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=use_bias,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format='channels_last')

def batch_norm(x, is_training):
    return tf.layers.batch_normalization(
        x, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
        training=is_training, fused=True)

def bottleneck_block_v2(x, filters, is_training, strides, projection=False):
    shortcut = x
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    if projection: shortcut = conv2d_layer(x, filters * 4, 1, strides)
    x = conv2d_layer(x, filters, 1, 1)
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    x = conv2d_layer(x, filters, 3, strides)
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    x = conv2d_layer(x, filters * 4, 1, 1)
    return x + shortcut

def block_layer(x, is_training, filters, num_blocks, strides):
    x = bottleneck_block_v2(x, filters, is_training, strides, projection=True)
    for _ in range(1, num_blocks):
        x = bottleneck_block_v2(x, filters, is_training, 1)
    return x

def vgg_layer(x, is_training, filters, num_blocks, pooling=True):
    for _ in range(num_blocks):
        x = conv2d_layer(x, filters, 3, 1)
        x = tf.nn.relu(batch_norm(x, is_training))
    if pooling:
        x = tf.layers.max_pooling2d(x, 2, 2, padding='VALID')
    return x 

def flatten_layer(x):
    sh = x.shape
    x = tf.reshape(x, [-1, sh[1] * sh[2] * sh[3]])
    return x

def tum(x, is_training, scales):
    branch = [x]
    for i in range(scales - 1):
        without_padding = True if np.min(x.shape[1:3]) <= 3 else False
        x = conv2d_layer(x, filters=256, kernel_size=3, strides=2, without_padding=without_padding)
        x = tf.nn.relu(batch_norm(x, is_training))
        branch.insert(0, x)
    out = [x]
    for i in range(1, scales):
        x = conv2d_layer(x, filters=256, kernel_size=3, strides=1)
        x = tf.nn.relu(batch_norm(x, is_training))
        x = tf.image.resize_images(x, tf.shape(branch[i])[1:3], method=tf.image.ResizeMethod.BILINEAR)
        x = x + branch[i]
        out.append(x)
    for i in range(scales):
        out[i] = conv2d_layer(out[i], filters=128, kernel_size=1, strides=1)
        out[i] = tf.nn.relu(batch_norm(out[i], is_training))
    return out
