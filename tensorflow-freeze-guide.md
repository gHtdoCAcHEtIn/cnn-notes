# How to process TF/Slim models

## Step 1 - Download models

All models were downloaded from tf/slim library
[github/tensorflow/models/research/slim](https://github.com/tensorflow/models/tree/master/research/slim)

* inception_resnet_v2_2016_08_30.tar.gz
* inception_v1_2016_08_28.tar.gz
* inception_v3_2016_08_28.tar.gz
* resnet_v1_101_2016_08_28.tar.gz
* resnet_v1_50_2016_08_28.tar.gz
* vgg_16_2016_08_28.tar.gz
* vgg_19_2016_08_28.tar.gz
* etc.

## Step 2 - Untar

To see the contents of the tar file

```bash
tar -tf inception_v3_2016_08_28.tar.gz
```

To extract the `CHECKPOINT` files in `.ckpt` format:

```bash
tar -xzvf inception_v3_2016_08_28.tar.gz
```

## Step 3 - Get inference graphs

Inside a `tensorflow/tensorflow:1.8.0-devel-gpu` container, go to tf-models/research/slim directory, and run `export_inference_graph.py` like this:

```bash
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inference_graph.pb
```

Place the output inference graph in your working directory.

## Step 4 - Figure out input and output nodes of the graph

Inside a `tensorflow/tensorflow:1.8.0-devel-gpu` container, go to `/tensorflow` directory, and run the following

```bash
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph /tmp/inception_v3_inference_graph.pb
```

This will print a lot of text. Filter out `--output_layer` for output nodes. For `inception_v3`, the output node is `InceptionV3/Predictions/Reshape_1`.
Here's a list for popular networks incase:

```json
inception_v1: InceptionV1/Logits/Predictions/Reshape_1
inception_v3: InceptionV3/Predictions/Reshape_1
resnet_v1_50: resnet_v1_50/predictions/Reshape_1
resnet_v1_101: resnet_v1_101/predictions/Reshape_1
inception_resnet_v2: InceptionResnetV2/Logits/Predictions
vgg16: vgg_16/fc8/squeezed
vgg19: vgg_19/fc8/squeezed
lenet: Predictions/Reshape_1
```

## Step 5 - Freeze the inference graph

Inside a `tensorflow/tensorflow:1.8.0-devel-gpu` container, check if `which freeze_graph` exists. If exists, skip the build step below. If not, build like this in `/tensorflow`:

```bash
bazel build tensorflow/python/tools:freeze_graph
```

Once built, run the binary `freeze_graph` like this:

```bash
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```

## Step 6 - Success

Go sleep!
