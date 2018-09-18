# Benchmark TensorRT optimizations

TensorRT comes with a `trtexec` sample application to quickly run and test `caffe` and `tensorflow` models.
Here's the usage instruction for the app.

```sh
$ ./bin/trtexec

Mandatory params:
  --deploy=<file>      Caffe deploy file
  OR --uff=<file>      UFF file
  --output=<name>      Output blob name (can be specified multiple times)

Mandatory params for onnx:
  --onnx=<file>        ONNX Model file

Optional params:
  --uffInput=<name>,C,H,W Input blob names along with their dimensions for UFF parser
  --model=<file>       Caffe model file (default = no model, random weights used)
  --batch=N            Set batch size (default = 1)
  --device=N           Set cuda device to N (default = 0)
  --iterations=N       Run N iterations (default = 10)
  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=10)
  --percentile=P       For each iteration, report the percentile time at P percentage (0<P<=100, default = 99.0%)
  --workspace=N        Set workspace size in megabytes (default = 16)
  --fp16               Run in fp16 mode (default = false). Permits 16-bit kernels
  --int8               Run in int8 mode (default = false). Currently no support for ONNX model.
  --verbose            Use verbose logging (default = false)
  --hostTime           Measure host time rather than GPU time (default = false)
  --engine=<file>      Generate a serialized TensorRT engine
  --calib=<file>       Read INT8 calibration cache file.  Currently no support for ONNX model.
```

This page has two sections
* [Tensorflow](#tensorflow)
* [Caffe](#caffe)

## Tensorflow
### Step 1: Convert Tensorflow frozen graph models to UFF format
* Make sure to install both tensorflow, and tensorrt (python packages: tensorrt, uff, graphsurgeon) for this to work.
* Run `convert-to-uff` utility that ships with python package `uff` to convert frozen graph models to UFF format

```bash
$ convert-to-uff tensorflow \
    -o <output.uff> \
    --input-file <frozen-graph.pb> \
    -O <output-tensor-name>
```
Here are examples for LeNet, Inception_v1, and Resnet_v1_50 models.

LeNet:

```bash
$ convert-to-uff tensorflow \
  -o lenet.uff \
  --input-file lenet_frozen_graph.pb \
  -O Predictions/Reshape_1
```

Inception_V1:

```bash
$ convert-to-uff tensorflow \
  -o inception_v1.uff \
  --input-file inception_v1_frozen_graph.pb \
  -O InceptionV1/Logits/Predictions/Reshape_1
```

Resnet_v1_50:

```bash
$ convert-to-uff tensorflow \
  -o resnet_v1_50.uff \
  --input-file resnet_v1_50_frozen_graph.pb \
  -O resnet_v1_50/predictions/Reshape_1
```

### Step 2: Run UFF model files with TensorRT utility `trtexec`

This utility is installed with TensorRT installation (refer to Nvidia's TensorRT installation guide)

`TRTEXEC` usage with UFF files

```bash
$ ./bin/trtexec \
    --uff=<model.uff> \
    --output=<output-node> \
    --uffInput=<input-tensor>,<num-channels>,<image-height>,<image-width>
```

#### Results

Here are examples for using `trtexec` with LeNet, Inception_v1, and ResNet_v1_50 models

LeNet

```bash
$ ./bin/trtexec --uff=lenet.uff --output="Predictions/Reshape_1" \
    --uffInput=input,1,28,28

uff: lenet.uff
output: Predictions/Reshape_1
uffInput: input,1,28,28
name=input, bindingIndex=0, buffers.size()=2
name=Predictions/Reshape_1, bindingIndex=1, buffers.size()=2
Average over 10 runs is 0.18319 ms (percentile time is 0.366592).
Average over 10 runs is 0.158928 ms (percentile time is 0.165888).
Average over 10 runs is 0.159846 ms (percentile time is 0.19968).
Average over 10 runs is 0.157392 ms (percentile time is 0.178176).
Average over 10 runs is 0.168653 ms (percentile time is 0.232448).
Average over 10 runs is 0.157594 ms (percentile time is 0.16384).
Average over 10 runs is 0.159027 ms (percentile time is 0.173056).
Average over 10 runs is 0.165478 ms (percentile time is 0.211968).
Average over 10 runs is 0.158211 ms (percentile time is 0.162816).
Average over 10 runs is 0.157898 ms (percentile time is 0.162816).
```

Inception_v1

```bash
$ ./bin/trtexec --uff=inception_v1.uff --output=InceptionV1/Logits/Predictions/Reshape_1 \
    --uffInput=input,3,224,224

uff: inception_v1.uff
output: InceptionV1/Logits/Predictions/Reshape_1
uffInput: input,3,224,224
name=input, bindingIndex=0, buffers.size()=2
name=InceptionV1/Logits/Predictions/Reshape_1, bindingIndex=1, buffers.size()=2
Average over 10 runs is 1.45111 ms (percentile time is 1.55034).
Average over 10 runs is 1.44282 ms (percentile time is 1.4633).
Average over 10 runs is 1.44517 ms (percentile time is 1.48378).
Average over 10 runs is 1.43943 ms (percentile time is 1.45306).
Average over 10 runs is 1.42643 ms (percentile time is 1.45306).
Average over 10 runs is 1.43636 ms (percentile time is 1.45306).
Average over 10 runs is 1.44046 ms (percentile time is 1.46125).
Average over 10 runs is 1.44671 ms (percentile time is 1.46227).
Average over 10 runs is 1.43933 ms (percentile time is 1.4551).
Average over 10 runs is 1.33304 ms (percentile time is 1.34451).
```

Resnet_v1_50

```bash
$ ./bin/trtexec --uff=resnet_v1_50.uff --output=resnet_v1_50/predictions/Reshape_1 \
    --uffInput=input,3,224,224
  
uff: resnet_v1_50.uff
output: resnet_v1_50/predictions/Reshape_1
uffInput: input,3,224,224
name=input, bindingIndex=0, buffers.size()=2
name=resnet_v1_50/predictions/Reshape_1, bindingIndex=1, buffers.size()=2
Average over 10 runs is 3.2642 ms (percentile time is 3.40992).
Average over 10 runs is 3.23932 ms (percentile time is 3.26451).
Average over 10 runs is 3.24024 ms (percentile time is 3.25325).
Average over 10 runs is 3.2426 ms (percentile time is 3.26246).
Average over 10 runs is 3.12627 ms (percentile time is 3.2512).
Average over 10 runs is 3.00616 ms (percentile time is 3.02182).
Average over 10 runs is 2.99356 ms (percentile time is 3.00544).
Average over 10 runs is 2.99592 ms (percentile time is 3.00442).
Average over 10 runs is 2.99295 ms (percentile time is 3.00646).
Average over 10 runs is 2.99336 ms (percentile time is 2.9993).
```


#### Known errors

Use of `--uffInput` is necessary, otherwise the programs errors out like this:

```bash
$ ./bin/trtexec --uff=inception_v1.uff --output=InceptionV1/Logits/Predictions/Reshape_1

uff: inception_v1.uff
output: InceptionV1/Logits/Predictions/Reshape_1
Parameter check failed at: ../builder/Network.cpp::addInput::364, condition: isValidDims(dims)
Segmentation fault (core dumped)
```

## Caffe

Caffe (`.prototxt` and `.caffemodel`) models can be directly used with `trtexec`

Usage:
```bash
$ ./bin/trtexec --deploy=<model>.prototxt --ouput=<output-blob>

$ ./bin/trtexec --deploy=resnet50.prototxt --ouput=prob
```


### Results

Here are results with Caffe and TensorRT

LeNet

```bash
```

ResNet_v1_50

```bash
$ ./bin/trtexec --deploy=resnet50.prototxt --output=prob

deploy: deploy.prototxt
output: prob
Input "data": 3x224x224
Output "prob": 1000x1x1
name=data, bindingIndex=0, buffers.size()=2
name=prob, bindingIndex=1, buffers.size()=2
Average over 10 runs is 3.06995 ms (percentile time is 3.08531).
Average over 10 runs is 3.06616 ms (percentile time is 3.08941).
Average over 10 runs is 3.06872 ms (percentile time is 3.08736).
Average over 10 runs is 3.07558 ms (percentile time is 3.09043).
Average over 10 runs is 2.88604 ms (percentile time is 3.05971).
Average over 10 runs is 2.84908 ms (percentile time is 2.86413).
Average over 10 runs is 2.85297 ms (percentile time is 2.87232).
Average over 10 runs is 2.8417 ms (percentile time is 2.86208).
Average over 10 runs is 2.83208 ms (percentile time is 2.85184).
Average over 10 runs is 2.82829 ms (percentile time is 2.83853).

```

Inception_v1

```bash
$ ./bin/trtexec --deploy=deploy2.prototxt --output=prob

deploy: deploy2.prototxt
output: prob
Input "data": 3x299x299
Output "prob": 1000x1x1
name=data, bindingIndex=0, buffers.size()=2
name=prob, bindingIndex=1, buffers.size()=2
Average over 10 runs is 6.75001 ms (percentile time is 6.83315).
Average over 10 runs is 6.73567 ms (percentile time is 6.76557).
Average over 10 runs is 6.4043 ms (percentile time is 6.75123).
Average over 10 runs is 6.24312 ms (percentile time is 6.29146).
Average over 10 runs is 6.238 ms (percentile time is 6.27098).
Average over 10 runs is 6.22019 ms (percentile time is 6.25459).
Average over 10 runs is 6.21261 ms (percentile time is 6.24026).
Average over 10 runs is 6.21793 ms (percentile time is 6.25766).
Average over 10 runs is 6.20882 ms (percentile time is 6.24742).
Average over 10 runs is 6.21619 ms (percentile time is 6.29248).

$ ./bin/trtexec --deploy=deploy.prototxt --output=prob

deploy: deploy.prototxt
output: prob
Input "data": 3x299x299
Output "prob": 1000x1x1
name=data, bindingIndex=0, buffers.size()=2
name=prob, bindingIndex=1, buffers.size()=2
Average over 10 runs is 7.75014 ms (percentile time is 7.88787).
Average over 10 runs is 7.65675 ms (percentile time is 7.7609).
Average over 10 runs is 7.33501 ms (percentile time is 7.73632).
Average over 10 runs is 7.11178 ms (percentile time is 7.17312).
Average over 10 runs is 7.54227 ms (percentile time is 9.16378).
Average over 10 runs is 7.11721 ms (percentile time is 7.52947).
Average over 10 runs is 7.01686 ms (percentile time is 7.15264).
Average over 10 runs is 6.99832 ms (percentile time is 7.05638).
Average over 10 runs is 7.00938 ms (percentile time is 7.05331).
Average over 10 runs is 7.04481 ms (percentile time is 7.09222).
```
