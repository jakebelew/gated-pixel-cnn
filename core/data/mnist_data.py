import collections

import numpy as np
import tensorflow as tf

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'height', 'width', 'channels'])


def _preprocess_dataset(dataset, preprocess_fcn, dtype=tf.float32, reshape=True):
  from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
  images, labels = preprocess_fcn(dataset.images, dataset.labels)    
  return DataSet(images, labels, dtype, reshape)

  
def _colorize(preprocess_fcn=None):
  
  def colorize_fcn(images, labels):
    num_images = images.shape[0]
    num_rgb_channels = 3
    num_exclude = np.random.randint(num_rgb_channels, size=num_images)
    exclude_channels = [np.sort(np.random.choice(num_rgb_channels, ne, replace=False)) for ne in num_exclude]  
    rgb_images = np.repeat(images, num_rgb_channels, axis=3)
    for i, ec in enumerate(exclude_channels):
      rgb_images[i, :, :, ec] = 0
  
    if preprocess_fcn is not None:
      rgb_images, labels = preprocess_fcn(rgb_images, labels)  
    return (rgb_images, labels)
  
  return colorize_fcn


def get_dataset(data_dir, preprocess_fcn=None, dtype=tf.float32, reshape=True):
  """Construct a DataSet.
  `dtype` can be either
  `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
  `[0, 1]`.
   `reshape` Convert shape from [num examples, rows, columns, depth]
    to [num examples, rows*columns] (assuming depth == 1)    
  """
  from tensorflow.examples.tutorials.mnist import input_data

  datasets = input_data.read_data_sets(data_dir, dtype=dtype, reshape=reshape)
  
  if preprocess_fcn is not None:
    train = _preprocess_dataset(datasets.train, preprocess_fcn, dtype, reshape)
    validation = _preprocess_dataset(datasets.validation, preprocess_fcn, dtype, reshape)
    test = _preprocess_dataset(datasets.test, preprocess_fcn, dtype, reshape)
  else:
    train = datasets.train
    validation = datasets.validation
    test = datasets.test

  height, width, channels = 28, 28, 1 
  return Datasets(train, validation, test, height, width, channels)


def get_colorized_dataset(data_dir, preprocess_fcn=None, dtype=tf.float32, reshape=True):
  datasets = get_dataset(data_dir, _colorize(preprocess_fcn), dtype, reshape)
  channels = 3
  return Datasets(datasets.train, datasets.validation, datasets.test, datasets.height, datasets.width, channels)
