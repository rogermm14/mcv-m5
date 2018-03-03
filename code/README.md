# About how to run the code

#### Object recognition

  - VGG 
 
    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif.py -e vgg_default -l /home/master/ -s /data/module5
    ```
    
    - Substract mean and normalize by standard deviation (computed from training set)  [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif_preprocess.py -e vgg_preprocessing -l /home/master/ -s /data/module5
    ```
    
    - Use random crops of 224x224 to feed the net [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif_crops.py -e vgg_input_crops -l /home/master/ -s /data/module5
    ```
    
    - Use resized images of 224x224 to feed the net [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif_resize.py -e vgg_input_resize -l /home/master/ -s /data/module5
    ```
    
