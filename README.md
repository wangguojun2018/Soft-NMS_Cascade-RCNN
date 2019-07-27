# **Soft-NMS_Cascade-RCNN**

## Abstract
This is a tensorflow re-implementation of [Cascade R-CNN Delving into High Quality Object Detection ](https://arxiv.org/abs/1712.00726).       

This project is based on Cascade-RCNN_Tensorflow by [YangXue](https://github.com/yangxue0827) and [WangYashan](https://github.com/toubasi).   This project fixed the bug about MobilenetV2 and add Vgg16 backbone. Besides Soft-Nms are implemented 

## Train on VOC 2007 trainval and test on VOC 2007 test (PS. This project also support coco training.)     
## Comparison
### use_voc2007_metric

| method | AP | AP50 | AP60 | AP70 | AP80 | AP90 |
|------------|:---:|:--:|:--:|:--:|:--:|:--:|
|Vgg16 Faster-RCNN|0.4059| 0.6954 | 0.6032 | 0.4506 | 0.2319 | 0.0488 |
|Soft-NMS Vgg16 Faster-RCNN|||||||
|Vgg16 Cascade-RCNN|0.4468| 0.6856 | 0.6120 | 0.4951 | 0.3331 | 0.1080 |
|MobilenetV2-Faster-RCNN|        | 0.5416 |        |        |        |        |
|Soft-NMS MobilenetV2-Faster-RCNN|        | 0.5423 |        |        |        |        |

### 
## Requirements
1、tensorflow >= 1.2     
2、cuda8.0     
3、python2.7 (anaconda2 recommend)    
4、[opencv(cv2)](https://pypi.org/project/opencv-python/)    

## Download Model
1、please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) 、

[vgg16]: http://arxiv.org/abs/1409.1556.pdf

 pre-trained models on Imagenet, put it to $PATH_ROOT/data/pretrained_weights.     
2、please download [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) pre-trained model on Imagenet, put it to $PATH_ROOT/data/pretrained_weights/mobilenet.     
3、please download [trained model](https://github.com/DetectionTeamUCAS/Models/tree/master/Cascade-RCNN_Tensorflow) by this project, put it to $PATH_ROOT/output/trained_weights.   

## Data Format
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│   ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```

## Demo

**Select a configuration file in the folder ($PATH_ROOT/libs/configs/) and copy its contents into cfgs.py, then download the corresponding [weights](https://github.com/DetectionTeamUCAS/Models/tree/master/Cascade-RCNN_Tensorflow).**      

```   
cd $PATH_ROOT/tools
python inference.py --data_dir='/PATH/TO/IMAGES/' 
                    --save_dir='/PATH/TO/SAVE/RESULTS/' 
                    --GPU='0'
```

## Eval
```  
cd $PATH_ROOT/tools
python eval.py --eval_imgs='/PATH/TO/IMAGES/'  
               --annotation_dir='/PATH/TO/TEST/ANNOTATION/'
               --GPU='0'
```



## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection

4、https://github.com/DetectionTeamUCAS/Cascade-RCNN_Tensorflow.git