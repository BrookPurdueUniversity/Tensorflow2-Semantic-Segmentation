
# Tensorflow2-Semantic-Segmentation

* Brook Cheng
* July 11 2021

## Introduction
Transfer the code files of Pix2Pix semantic segmentation from tf.__version_=1.13.1 to tf.__version__=2.3.0

## Dependencies
* tensorflow2.3.0
* tf_slim
* numpy
* opencv
* matplotlib
* sklearn

## Switch 

#### I.  tf_1 has tf.contrib, tf_2 removed tf.contrib
* tf_1: import tensorflow.contrib.slim as slim
* tf2_2: import tf_slim as slim  (pip install --upgrade tf_slim)

#### II.  tensorflow in v2
* tf_1: import tensorflow as tf
* tf2_2: import tensorflow.compat.v1 as tf

#### III.  
* tf_1: tf.reset_default_graph()
* tf_2: tf.compat.v1.reset_default_graph()

#### IV.  
* tf_1: net_input = tf.placeholder(tf.float32,shape=[None,None,None,3]) 
* tf_2: tf.compat.v1.disable_eager_execution()
* net_input = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,3])

#### V. 
* tf_1: net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
* tf_2: net_output = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,num_classes])

#### VI.
* tf_1:// Allow GPU for tensorflow < 2.0
* config = tf.ConfigProto()   
* config.gpu_options.allow_growth = True
* sess=tf.Session(config=config)

#### VII.
* tf_1:// Allow GPU for tensorflow >= 2.0
* config = tf.compat.v1.ConfigProto()
* config.gpu_options.allow_growth=True
* sess = tf.compat.v1.Session(config=config)


```python

```
