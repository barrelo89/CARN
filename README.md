# CARN in Tensorflow
This is the implementation of an ECCV'18 paper, "Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network" in Tensorflow [(link)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Namhyuk_Ahn_Fast_Accurate_and_ECCV_2018_paper.pdf)

## Prerequisite
- Language: Python3
- Required Packages: numpy, cv2 (for read image input), and tensorflow
- In this implementation, tensorflow-gpu==2.0.0 is installed. If higher version of TF used, you can easily enable 'group convolution' in tf.keras.layers.Conv2D by setting 'groups' attribute to the number of groups you want 
- To install the required packages, type the following command:

1) Using tensorflow-gpu==2.0.0
```
pip3 install numpy opencv-python tensorflow-gpu==2.0.0
```

2) Using higher version of tensorflow
```
pip3 install numpy opencv-python tensorflow
```


