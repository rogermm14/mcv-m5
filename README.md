# Scene Understanding for Autonomous Vehicles
Master in Computer Vision, Barcelona (2017-2018) - M5 Visual Recognition

## About us
Name of the group: TEAM 03 (a.k.a. The Young Team)  
[Roger Marí](https://github.com/rogermm14). Email: roger.mari01@estudiant.upf.edu  
[Joan Sintes](https://github.com/JoSintes8). Email: joan.sintes01@estudiant.upf.edu  
[Àlex Palomo](https://github.com/alexpalomodominguez). Email: alex.palomo01@estudiant.upf.edu  
[Àlex Vicente](https://github.com/AlexVicenteS). Email: alex.vicente01@estudiant.upf.edu  

## Abstract
This 5-week project presents a series of experiments related to scene understanding for autonomous vehicles.   
Deep learning is employed to face 3 main tasks: object recognition, object detection and semantic segmentation.  

## Project report
You can check our Overleaf report [here](https://www.overleaf.com/read/mgdfttmpqkgx).

## Project slides
You can check our slides [here](https://docs.google.com/presentation/d/1Vlk9INjR2pFve4IUYKt027kSwZSVRazxz6rFk_DsciM/edit?usp=sharing).

## Weights of the trained architectures
You can download the weights associated to each experiment [here](https://drive.google.com/open?id=1E-GTzCvu4uPF5l0tL_0v8Tz-P6YeWMn2). Experiment names and description can be found [here](https://github.com/rogermm14/mcv-m5/blob/master/code/README.md). 

## Weeks 1-2. Object recognition
Evaluation and comparison of the performance of the VGG-16 and the ResNet-50 architectures for object recogniton.    
The TSingHua-TenCent 100K ([TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)) dataset **[3]** and the Belgium Traffic Sign ([BelgiumTS](http://btsd.ethz.ch/shareddata/)) dataset **[4]** are used to train for traffic sign recognition. The [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) dataset **[5]** is used to train for recognition of vehicles, pedestrians and cyclists.    
See this [README](https://github.com/rogermm14/mcv-m5/blob/master/code/README.md) to gain further insight about how to run the code for the different experiments.   
You can find our summary of the VGG paper [here](https://www.overleaf.com/read/bpwcjjmpnnsy) **[1]**.   
Also, find our summary of the ResNet paper [here](https://www.overleaf.com/read/qwdjmppkrpcg) **[2]**.   

#### Completed tasks:

1. **Train VGG using TT100K dataset**   
  - [x] Train from scratch.    
  - [x] Analyze train/validation/test sets and interpret results.    
  - [x] Comparison between crop and resize to feed the net.   
  - [x] Transfer learning to BelgiumTS dataset.   
2. **Train VGG using KITTI dataset**   
  - [x] Train from scratch.    
  - [x] Fine-tunning based on the ImageNet weights **[6]**.    
3. **Train ResNet using TT100K dataset**    
  - [x] Implementation of ResNet-50 with Keras and integration to the framework.       
  - [x] Train from scratch.    
  - [x] Fine-tunning based on the ImageNet weights.  
4. **Boost perfromance of the network**    
  - [x] Retrain all layers of ResNet-50 using ImageNet weights as initialization.      
  - [x] Use data augmentation to boost the performance.       
  - [x] Experiment with a mixed architecture (conv. net from ResNet-50 + fully connected layers from VGG-16).

#### Contributions to the code:    
+ `code/models/resnet.py`. ResNet-50 implementation (*build_resnet50*) + mixed architecture (*build_resnet50_v2*).   
+ `other_scripts/analyze_dataset.py`. Analyzes the elements per class in the train/validation/test splits of a dataset.   
+ `other_scripts/run_experiments_week2.py`. Bash script to execute all experiments on object recognition.   
+ `code/config/*`. The configuration files of all the conducted experiments can be found in this folder.   

## Weeks 3-4. Object detection
Evaluation and comparison of the performance of the YOLO and the Faster R-CNN architectures for object detection.    
The TSingHua-TenCent 100K ([TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)) dataset (for detection) **[3]** is used to train for traffic sign detection, while the [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) annotated driving dataset is used to train for detection of pedestrians, cars and trucks.    
See this [README](https://github.com/rogermm14/mcv-m5/blob/master/code/README.md) to gain further insight about how to run the code for the different experiments.    
You can find our summary of the YOLO paper [here](https://www.overleaf.com/14784830gcsygvmrfzhg#/56350076/) **[7]**.    
Also, find our summary of the Faster R-CNN paper [here]() **[8]**.    

1. **Train YOLO using TT100K_detection dataset**   
  - [x] ImageNet weights for initialization + re-train all layers for 10 epochs.    
  - [x] Analyze train/validation/test sets and visualize/interpret results.     
  - [x] Compute F-score and FPS.   
2. **Train YOLO using Udacity dataset**   
  - [x] ImageNet weights for initialization + re-train all layers for 40 epochs.    
  - [x] Analyze train/validation/test sets and visualize/interpret results.     
  - [x] Compute F-score and FPS.       
3. **Train Faster R-CNN using TT100K_detection dataset**    
  - [x] Implementation of Faster R-CNN based on ResNet-50 with Keras and integration to the framework.      
  - [x] ImageNet weights for initialization + re-train all layers for 30 epochs.      
  - [x] Adjustment of detection threshold to keep a good compromise between Precision and Recall.
4. **Train Faster R-CNN using Udacity dataset**        
  - [x] ImageNet weights for initialization + re-train all layers for 30 epochs.      
  - [x] Adjustment of detection threshold to keep a good compromise between Precision and Recall.  
4. **Boost perfromance of the network**     
  - [x] Use data augmentation to boost the performance of YOLO.

#### Contributions to the code:    
+ `frcnn/*`. Faster R-CNN implementation as readapted from https://github.com/yhenon/keras-frcnn.  
+ `other_scripts/create_gt_frcnn_*.py`. Writes the labels of a dataset in the format required by the Faster R-CNN.
+ `other_scripts/evaluate_frcnn_*.py`. Computes F-score, Precision and Recall for Faster R-CNN.
+ `other_scripts/analyze_dataset_*.py`. Analyzes elements per class in the train/val/test splits of a given dataset
+ `other_scripts/run_experiments_week3.py`. Bash script to execute all experiments on object detection.   
+ `code/config/*`. The configuration files of all the conducted experiments can be found in this folder. 

## References

**[1]** Simonyan, Karen, and Andrew Zisserman. *Very deep convolutional networks for large-scale image recognition.* arXiv preprint arXiv:1409.1556 (2014).   
**[2]** He, Kaiming, et al. *Deep residual learning for image recognition.* Proceedings of the IEEE Conference on Computer vision and Pattern Recognition (CVPR). 2016.   
**[3]**  Zhu, Zhe, et al. *Traffic-sign detection and classification in the wild.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.   
**[4]** Timofte, Radu, Karel Zimmermann, and Luc Van Gool. *Multi-view traffic sign detection, recognition, and 3d localisation.* Machine Vision and Applications 25.3 (2014): 633-647.   
**[5]** Geiger, Andreas, Philip Lenz, and Raquel Urtasun. *Are we ready for autonomous driving? the kitti vision benchmark suite.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2012.   
**[6]** Deng, Jia, et al. *Imagenet: A large-scale hierarchical image database.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2009.    
**[7]** J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. *You only look once: Unified, real-time object detection.* arXiv preprint arXiv:1506.02640, 2015.   
**[8]** S. Ren, K. He, R. Girshick, and J. Sun. *Faster r-cnn: Towards real-time object detection with region proposal networks.* arXiv preprint arXiv:1506.01497, 2015.
