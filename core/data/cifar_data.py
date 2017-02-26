import collections

import numpy as np

import cifar10 as cifar

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'height', 'width', 'channels'])


# From github/tensorflow/tensorflow/tensorflow/contrib/learn/python/learn/datasets/mnist.py
class DataSet(object):

  def __init__(self,
               images,
               labels):
    """Construct a DataSet.
    Data is a ndarray of type float32 with values in the range [0., 1.]. 
    Images have shape [N,H,W,C] and labels have shape [N].
    """
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def get_dataset(data_dir, preprocess_fcn=None):
  """ train dataset has shape [50000, 32, 32, 3], test dataset has shape [10000, 32, 32, 3].
  """
  (X_train, Y_train), (X_test, Y_test) = cifar.load_data(data_dir)
  if preprocess_fcn is not None:
    X_train, Y_train = preprocess_fcn(X_train, Y_train)
    X_test, Y_test = preprocess_fcn(X_test, Y_test)

  train = DataSet(X_train, Y_train)
  test = DataSet(X_test, Y_test)
  height, width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
  
  return Datasets(train, test, height, width, channels)
'''
"""Construct input for CIFAR evaluation using the Reader ops.

Args:
  eval_data: bool, indicating if one should use the train or eval data set.
  data_dir: Path to the CIFAR-10 data directory.
  batch_size: Number of images per batch.

Returns:
  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
  labels: Labels. 1D tensor of [batch_size] size.
"""
'''