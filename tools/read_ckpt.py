import tensorflow as tf
import os
# from tensorflow.python import pywrap_tensorflow
# import tensorflow as tf

model_dir='../data/pretrained_weights'

# checkpoint_path=os.path.join(model_dir,'inception_v3.ckpt')
checkpoint_path='/home/wangguojun/source_code/py/Cascade-RCNN_Tensorflow/data/' \
                            'pretrained_weights/mobilenet/mobilenet_v2_1.0_224.ckpt'
# checkpoint_path=tf.train.latest_checkpoint('/home/wangguojun/source_code/py/Cascade-RCNN_Tensorflow/data/' \
#                             'pretrained_weights/mobilenet')
reader=tf.train.NewCheckpointReader(checkpoint_path)

var_to_shape_map=reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    if 'Logits' in key:
        print ('tensor_name: ',key,reader.get_tensor(key).shape)