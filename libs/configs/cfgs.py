# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
VERSION = 'FasterRCNN_20190724_v1_vgg16'
NET_NAME = 'vgg_16' #'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

#-----------------------------------------Dataset and preprocess config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
DATASET_DIR='/home/wangguojun/dataset/VOC2007/VOCdevkit'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
# PIXEL_MEAN=[122.7717, 115.9465,102.9801 ]
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000
CLASS_NUM = 20

MATLAB='matlab'

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 2
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000
VALIDATE_INTE=100

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
# INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
# INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
elif NET_NAME.startswith("vgg_16"):
    weights_name='vgg_16'
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2, vgg_16]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
PRETRAINED_MOBILE=ROOT_PATH+'/data/pretrained_weights/mobilenet'
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3
TRAIN_CONVS=[True,True,True,True] #vgg16中可训练的convs（1-4）
RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0


MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   #10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001 # 0.001  # 0.0003 0.002震荡很严重，导致不收敛
DECAY_STEP = [60000]  # 根据cascade 原文
MAX_ITERATION = 80000

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
# BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00005 if NET_NAME.startswith('Mobilenet') else 0.0005 #根据faster rcnn原文


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None


# --------------------------------------------RPN config
# KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 300  # 300


# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard



SOFT_NMS=True
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FAST_RCNN_IOU_POSITIVE_THRESHOLD = [0.5, 0.6, 0.7]
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 128  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = True

