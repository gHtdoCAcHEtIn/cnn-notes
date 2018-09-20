# Timing results with TensorRT

Timing results (in milliseconds or ms) for various Caffe and Tensorflow models, when run with TensorRT optimizations.

* Various batchsizes are tried (1, 10, 100)
* Various precision values are tried (fp32 = default, fp16, and int8)

## CAFFE

                batchsize (fp32)            batchsize=100
models          1       10      100         fp16    int8

lenet           0.11    0.18    0.22        0.19    0.15
vgg16           2.9     16.     160.        41.2    62.4
resnet18        1.5     3.1     23.9        8.1     8.3
resnet50        2.8     7.7     59.1        19.2    18.5
googlenet       1.44    4.48    26.7        12.49   9.65
inceptionV3     7.1     16.2    111.8       53.4    31.48

## TENSORFLOW

                batchsize (fp32)            batchsize=100
models          1       10      100         fp16    int8

lenet+          0.14    0.25    0.44        0.23     0.29
vgg16           4.2     18.8    170.9       45.5     67.4
vgg19           4.6     21.9    204.8       50.8     77.9
resnet50        3.3     8.0     63.7        22.3     19.4
resnet101       5.6     14.4    120.0       39.7     38.0  
inception_v1    1.44    4.1     26.7        14.4     9.61
inception_v3*   6.2     14.5    97.         47.3     28.8

All networks take a 3x224x224 uffInput
+takes a 1x28x28 image
*takes a 3x299x299 image
