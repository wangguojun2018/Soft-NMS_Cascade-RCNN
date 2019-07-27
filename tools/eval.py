# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.networks import build_faster_rcnn
from libs.box_utils import draw_box_in_img
import argparse
from help_utils import tools
from data.lib_pascal.pascal_voc import pascal_voc
from data.lib_pascal.factory import get_imdb
from libs.box_utils.cython_utils.soft_nms import py_soft_nms
# from libs.nms.gpu_nms import gpu_nms

def eval_with_plac(det_net, real_test_imgname_list, draw_imgs=False):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    restorer, restore_ckpt = det_net.get_restorer()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        all_boxes = []
        for i, a_img_name in enumerate(real_test_imgname_list):

            raw_img = cv2.imread(a_img_name)
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            end = time.time()
            if cfgs.SOFT_NMS:
                boxes_soft_nms=[]
                scores_soft_nms=[]
                category_soft_nms=[]
                tmp_boxes=np.reshape(detected_boxes,(cfgs.CLASS_NUM,-1,4),order='C')
                # print("tmp_boxes is ", tmp_boxes.shape, 'type is ', type(tmp_boxes))
                tmp_scores=np.reshape(detected_scores,(cfgs.CLASS_NUM,-1),order='C')
                tmp_category=np.reshape(detected_categories,(cfgs.CLASS_NUM,-1),order='C')
                for ind in range(cfgs.CLASS_NUM):
                    tmp_class_boxes=tmp_boxes[ind,:,:]
                    tmp_class_scores=tmp_scores[ind,:]
                    tmp_class_category=tmp_category[ind,:]
                    # print("tmp_boxes is ",tmp_class_boxes.shape,'type is ',type(tmp_class_boxes))
                    # print('tmp score is ',tmp_class_scores.shape)
                    dets=np.hstack((tmp_class_boxes,tmp_class_scores[:,np.newaxis])).astype(np.float32,copy=False)
                    # keep=py_cpu_softnms(tmp_class_boxes,tmp_class_scores,method=4)
                    keep=py_soft_nms(dets,'greedy')
                    # keep=gpu_nms(dets,cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,0)
                    class_boxes=tmp_class_boxes[keep]
                    class_scores=tmp_class_scores[keep]
                    class_category=tmp_class_category[keep]
                    boxes_soft_nms.append(class_boxes)
                    scores_soft_nms.append(class_scores)
                    category_soft_nms.append(class_category)
                detected_boxes=np.concatenate(boxes_soft_nms,axis=0)
                detected_scores=np.concatenate(scores_soft_nms,axis=0)
                detected_categories=np.concatenate(category_soft_nms,axis=0)

            # print("{} cost time : {} ".format(img_name, (end - start)))
            if draw_imgs:
                show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores)
                if not os.path.exists(cfgs.TEST_SAVE_PATH):
                    os.makedirs(cfgs.TEST_SAVE_PATH)

                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/' + a_img_name + '.jpg',
                            final_detections[:, :, ::-1])

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
            dets = np.hstack((detected_categories.reshape(-1, 1),
                              detected_scores.reshape(-1, 1),
                              boxes))
            all_boxes.append(dets)

            tools.view_bar('{} image cost {}s'.format(os.path.basename(a_img_name), (end - start)), i + 1, len(real_test_imgname_list))

        save_dir = os.path.join(cfgs.EVALUATE_DIR,cfgs.DATASET_NAME, cfgs.VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fw1 = open(os.path.join(save_dir, 'detections.pkl'), 'wb')
        pickle.dump(all_boxes, fw1)
        return all_boxes


def eval(eval_dataset, showbox):


    test_image_index = eval_dataset.image_index
    test_image_list = [eval_dataset.image_path_from_index(index) for index in test_image_index]

    faster_rcnn = build_faster_rcnn.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    all_boxes=eval_with_plac(det_net=faster_rcnn, real_test_imgname_list=test_image_list,draw_imgs=showbox)

    save_dir = os.path.join(cfgs.EVALUATE_DIR,cfgs.DATASET_NAME, cfgs.VERSION)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(os.path.join(save_dir, 'detections.pkl'), 'rb') as f:
    #     all_boxes = pickle.load(f)

    print('检测boxes数量为： ',len(all_boxes))
    eval_dataset.evaluate_detections(all_boxes,save_dir)

def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')

    parser.add_argument('--dataset', dest='dataset',
                        help='evaluate dataset ',
                        default='voc_2007_test', type=str)
    parser.add_argument('--showbox', dest='showbox',
                        help='whether show detecion results when evaluation',
                        default=False, type=bool)
    parser.add_argument('--GPU', dest='GPU',
                        help='gpu id',
                        default='0', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':

    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    dataset=get_imdb(args.dataset)
    eval(eval_dataset=dataset,showbox=args.showbox)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # eval(num_imgs=np.inf,  # use np.inf to test all the imgs. use 10 to test 10 imgs.
    #      eval_dir='/data/VOC/VOC_test/VOC2007/JPEGImages/',
    #      annotation_dir='/data/VOC/VOC_test/VOC2007/Annotations',
    #      showbox=False)
















