#coding=utf-8
from libs.networks import build_faster_rcnn
import tensorflow as tf
from libs.configs import cfgs
from tensorflow.contrib import  slim as slim
from tensorflow.contrib.slim.nets import vgg


faster_rcnn=build_faster_rcnn.DetectionNetwork(base_network_name=cfgs.NET_NAME,is_training=False)


img_batch=tf.ones([1,448,448,3],dtype=tf.float32)
gt_batch=tf.ones([1,5],dtype=tf.float32)

weights_regularizer=slim.l2_regularizer(0.005)
biases_regularizer=tf.no_regularizer
with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
    final_score,final_boxes,final_category=faster_rcnn.build_whole_detection_network(img_batch,gt_batch)
var_lists=tf.trainable_variables()
model_var=slim.get_model_variables()

init=tf.global_variables_initializer()
checkpointpath='/home/wangguojun/source_code/py/' \
               'Cascade-RCNN_Tensorflow/data/pretrained_weights/mobilenet/mobilenet_v2_1.0_224.ckpt'

restore_var_list={}
for var in model_var:
    if var.name.startswith('Mobilenet'):
    # if var.name.startswith(cfgs.NET_NAME):
        restore_var_list[var.op.name]=var
    elif var.name.startswith('Fast-RCNN/MobilenetV2'):
        var_in_ckpt='/'.join(var.op.name.split('/')[1:])
        restore_var_list[var_in_ckpt]=var
for key,value in restore_var_list.items():
    print ('model param is {}  {:15} '.format(value.name,str(value.get_shape())))
    print ('ckpt param is {} '.format(key))
restorer=tf.train.Saver(restore_var_list)
with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess,checkpointpath)
    sess.run(final_score)

