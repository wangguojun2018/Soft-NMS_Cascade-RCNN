#coding=utf-8



from libs.networks import inception_v3 as inception
import time
from datetime import datetime
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

def time_tensorflow_run(session,target,info_string,nums_batches=100):
    num_steps_burn_in=10
    total_duration=0.0
    total_duration_squared=0.0
    for i in range(nums_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target)
        duration=time.time()-start_time

        if i>=num_steps_burn_in:
            if not i %10:
                print ('%s: step %d, duration=%.3f' % (datetime.now(),i-num_steps_burn_in,duration))
            total_duration+=duration
            total_duration_squared+=duration**2
    mn=total_duration/nums_batches
    vr=total_duration_squared/nums_batches-mn**2
    sd=math.sqrt(vr)
    print ('%s: %s across %d steps,%.3f+/- %.3f sec /batch' % (datetime.now(),info_string,nums_batches,mn,sd))


minist=input_data.read_data_sets('/home/wangguojun/dataset/mnist')

print ('训练集大小为： ',minist.train.num_examples)

print ('验证集大小为： ',minist.validation.num_examples)

print ('测试集大小为： ',minist.test.num_examples)


batch_size=24
height,width=299,299
with tf.device('/gpu:0'):
    inputs=tf.random_uniform((batch_size,height,width,3))
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits,end_points=inception.inception_v3(inputs,is_training=False)
    init=tf.global_variables_initializer()
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(init)
        time_tensorflow_run(sess,logits,'Forward')