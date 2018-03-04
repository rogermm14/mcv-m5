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

## Week 1
You can find our summary of the VGG paper [here](https://www.overleaf.com/read/bpwcjjmpnnsy) **[1]**.
Also, find our summary of the ResNet paper [here](https://www.overleaf.com/read/qwdjmppkrpcg) **[2]**.

## Week 2
Evaluation and comparison of the performance of the VGG-16 and the ResNet-50 architectures for object recogniton.    
The TSingHua-TenCent 100K ([TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)) dataset **[3]** and the Belgium Traffic Sign ([BelgiumTS](http://btsd.ethz.ch/shareddata/)) dataset **[4]** are used to train for traffic sign recognition. The [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) dataset **[5]** is used to train for recognition of vehicles, pedestrians and cyclists.    
See this [README](https://github.com/rogermm14/mcv-m5/blob/master/code/README.md) to gain further insight about how to run the code for the different experiments.

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
+ `other_scripts/run_experiments.py`. Bash script to execute all experiments on object recognition.   
+ `code/config/*`. The configuration files of all the conducted experiments can be found in this folder.   


## References

**[1]** Simonyan, Karen, and Andrew Zisserman. *Very deep convolutional networks for large-scale image recognition.* arXiv preprint arXiv:1409.1556 (2014).   
**[2]** He, Kaiming, et al. *Deep residual learning for image recognition.* Proceedings of the IEEE Conference on Computer vision and Pattern Recognition (CVPR). 2016.   
**[3]**  Zhu, Zhe, et al. *Traffic-sign detection and classification in the wild.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.   
**[4]** Timofte, Radu, Karel Zimmermann, and Luc Van Gool. *Multi-view traffic sign detection, recognition, and 3d localisation.* Machine Vision and Applications 25.3 (2014): 633-647.   
**[5]** Geiger, Andreas, Philip Lenz, and Raquel Urtasun. *Are we ready for autonomous driving? the kitti vision benchmark suite.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2012.   
**[6]** Deng, Jia, et al. *Imagenet: A large-scale hierarchical image database.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2009.  
