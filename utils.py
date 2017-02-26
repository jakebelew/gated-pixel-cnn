import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import os
import sys
import urllib
import pprint
import tarfile
import tensorflow as tf

import datetime
import dateutil.tz
import numpy as np

import scipy.misc

pp = pprint.PrettyPrinter().pprint
logger = logging.getLogger(__name__)

def mprint(matrix, pivot=0.5):
  for array in matrix:
    print "".join("#" if i > pivot else " " for i in array)

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    total_count += int(count)
  logger.info("Total number of variables: %s" % "{:,}".format(total_count))

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')

def binarize(images):
  return (np.random.uniform(size=images.shape) < images).astype('float32')

def save_images(images, height, width, n_row, n_col, 
      cmin=0.0, cmax=1.0, directory="./", prefix="sample"):
  channels = images.shape[3]

  if channels == 1:
    images = images.reshape((n_row, n_col, height, width))
    images = images.transpose(0, 2, 1, 3)
    images = images.reshape((height * n_row, width * n_col))
  else:  
    images = images.reshape((n_row, n_col, height, width, channels))
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape((height * n_row, width * n_col, channels))
  
  filename = '%s_%s.jpg' % (prefix, get_timestamp())
  scipy.misc.toimage(images, cmin=cmin, cmax=cmax) \
      .save(os.path.join(directory, filename))

def get_model_dir(config, exceptions=None):
  attrs = config.__dict__['__flags']
  pp(attrs)

  keys = attrs.keys()
  keys.sort()
  keys.remove('data')
  keys = ['data'] + keys

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags

  for option, value in options.items():
    option = option.lower()

def check_and_create_dir(directory):
  if not os.path.exists(directory):
    logger.info('Creating directory: %s' % directory)
    os.makedirs(directory)
  else:
    logger.info('Skip creating directory: %s' % directory)
