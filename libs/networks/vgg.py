
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import vgg

trainable_convs=cfgs.TRAIN_CONVS
def vgg_arg_scope(is_training=True,weight_decay=cfgs.WEIGHT_DECAY,batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5, batch_norm_scale=True):
    # normlizer=slim.batch_norm()
    batch_norm_params={'decay':batch_norm_decay,'scale':batch_norm_scale,'epsilon':batch_norm_epsilon,
                      'updates_collections':tf.GraphKeys.UPDATE_OPS,'trainable':False}
    with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        trainable=is_training,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=batch_norm_params,
                        padding='SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as arg_sc:
            return arg_sc


def vgg16(img_batch,scope_name='vgg_16',is_training=True):
    with slim.arg_scope(vgg_arg_scope(is_training=is_training)):
        with tf.variable_scope(scope_name):
            net=slim.repeat(img_batch,2,slim.conv2d,64,[3,3],trainable=trainable_convs[0],scope='conv1')
            net=slim.max_pool2d(net,[2,2],padding='VALID',scope='pool1')
            net=slim.repeat(net,2,slim.conv2d,128,[3,3],trainable=trainable_convs[1],scope='conv2')
            net=slim.max_pool2d(net,[2,2],scope='pool2')
            net=slim.repeat(net,3,slim.conv2d,256,[3,3],trainable=trainable_convs[2],scope='conv3')
            net=slim.max_pool2d(net,[2,2],scope='pool3')
            net=slim.repeat(net,3,slim.conv2d,512,[3,3],trainable=trainable_convs[3],scope='conv4')
            net=slim.max_pool2d(net,[2,2],scope='pool4')
            net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')

    return net

