# Working with TF models

Tensorflow offers two types of saved models `CHECKPOINT` files and a `SavedModel`.
Source: [tf/checkpoints](https://www.tensorflow.org/guide/checkpoints)

* Checkpoints documentation [Tensorflow/Docs/Checkpoints](https://www.tensorflow.org/guide/checkpoints)
* SavedModel documentation [Tensorflow/Docs/SavedModel](https://www.tensorflow.org/guide/saved_model)

## Checkpoint
There are three checkpoint files, that is, `.data`, `.index`, `.meta` files.

### Meta graph
This is a protocol buffer which saves the complete Tensorflow graph; i.e. all variables, operations, collections etc. 
This file has `.meta` extension.

### Checkpoint file (<0.11)
This is a binary file which contains all the values of the weights, biases, gradients and all the other variables saved. 
This file has an extension `.ckpt`. 

So, to summarize : Tensorflow models before 0.11 contained only three files:

```
inception_v1.meta
inception_v1.ckpt
checkpoint
```

### Checkpoint file (>=0.11)
This is a binary file which contains all the values of the weights, biases, gradients and all the other variables saved. 
However, Tensorflow has changed this from version 0.11. Now, instead of single `.ckpt` file, we have two files:

```
mymodel.data-00000-of-00001
mymodel.index
```

`.data` file is the file that contains our training variables and we shall go after it.

Along with this, Tensorflow also has a file named `checkpoint` which simply keeps a record of latest checkpoint files saved.

So, to summarize, Tensorflow models for versions greater than 0.10 look like this:

```
checkpoint
inception_v1_model.data-00000-of-00001
inception_v1_model.index
inception_v1_model.meta
```

## Saving checkpoint files
To save Tensorflow variables in a session, create a `Saver` object before the session, and call the `save` member function while the session is alive.

```python
saver = tf.train.Saver()
```

Example:

```python
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model')
 
# This will save following files in Tensorflow v >= 0.11
# my_test_model.data-00000-of-00001
# my_test_model.index
# my_test_model.meta
# checkpoint
```

### Saver Options

* If we are saving the model after 1000 iterations, we shall call save by passing the step count:

```python
saver.save(sess, 'my_test_model', global_step=1000)
```

This will just append ‘-1000’ to the model name and following files will be created:

```
my_test_model-1000.index
my_test_model-1000.meta
my_test_model-1000.data-00000-of-00001
checkpoint
```

* Let’s say, while training, we are saving our model after every 1000 iterations, 
so .meta file is created the first time(on 1000th iteration) and we don’t need 
to recreate the .meta file each time(so, we don’t save the .meta file at 2000, 
3000.. or any other iteration). We only save the model for further iterations, 
as the graph will not change. Hence, when we don’t want to write the meta-graph we use this:

```python
saver.save(sess, 'my-model', global_step=step, write_meta_graph=False)
```

* If you want to keep only 4 latest models and want to save one model after every 2 
hours during training you can use max_to_keep and keep_checkpoint_every_n_hours like this.

```python
#saves a model every 2 hours and maximum 4 latest models are saved.
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
```

* Note, if we don’t specify anything in the tf.train.Saver(), it saves all the variables. 
What if, we don’t want to save all the variables and just some of them. We can specify the 
variables/collections we want to save. While creating the tf.train.Saver instance we pass 
it a list or a dictionary of variables that we want to save. Let’s look at an example:

```python
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1,w2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model', global_step=1000)
```

## Loading/Reading checkpoint files

To load a pretrained model, there are two things to be done.

* Create the network
You can create the network by writing python code to create each and every layer manually 
as the original model. However, if you think about it, we had saved the network in .meta 
file which we can use to recreate the network using tf.train.import() function like this: 
`saver = tf.train.import_meta_graph('my_test_model-1000.meta')`

Remember, import_meta_graph appends the network defined in .meta file to the current graph. 
So, this will create the graph/network for you but we still need to load the value of the 
parameters that we had trained on this graph.

* Load the weights

We can restore the parameters of the network by calling restore on this saver which is an 
instance of `tf.train.Saver()` class.

```python
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
```

After this, the value of tensors like w1 and w2 has been restored and can be accessed:

```python
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('my-model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))
##Model has been restored. Above statement will print the saved value of w1.
```

## Freezing Checkpoint files

For a temp training directory like this:

```
checkpoint
events.out.tfevents.1537203360.fc4e9a2930e6
events.out.tfevents.1537203745.fc4e9a2930e6
graph.pbtxt
model.ckpt-199334.data-00000-of-00001
model.ckpt-199334.index
model.ckpt-199334.meta
model.ckpt-266033.data-00000-of-00001
model.ckpt-266033.index
model.ckpt-266033.meta
model.ckpt-332756.data-00000-of-00001
model.ckpt-332756.index
model.ckpt-332756.meta
model.ckpt-399494.data-00000-of-00001
model.ckpt-399494.index
model.ckpt-399494.meta
model.ckpt-466313.data-00000-of-00001
model.ckpt-466313.index
model.ckpt-466313.meta

```

Say, we choose to freeze model # 399494. Copy the following files to a new directory like this:

```
graph.pbtxt
model.ckpt-399494.data-00000-of-00001
model.ckpt-399494.index
```

Specify the `ckpt` file to `freeze_graph` utility like the following. 
Note that we need only the `.index` and the `.data-<0>-of-<n>` file(s) as the checkpoint file. 

```bash
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=graph.pbtxt
  --input_checkpoint=model.ckpt-399494 \
  --input_binary=false \
  --output_graph=lenet_frozen_graph.pb \
  --output_node_names=Predictions/Reshape_1
```
Note that `input_binary` is `false`, since the graph is in a `.pbtxt` file and in ASCII format. If the graph was in binary format, `input_binary` should be set to `true`.


## References:
[Complete Tutorial to save and restore Tensorflow models](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
