import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
import os
import time

import numpy as np
import tensorflow as tf

import core.data.cifar_data as cifar
import core.data.mnist_data as mnist
from network import Network
from statistic import Statistic
import utils as util

flags = tf.app.flags

# network
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("gated_conv_num_layers", 7, "the number of gated conv layers")
flags.DEFINE_integer("gated_conv_num_feature_maps", 16, "the number of input / output feature maps in gated conv layers") 
flags.DEFINE_integer("output_conv_num_feature_maps", 32, "the number of output feature maps in output conv layers")
flags.DEFINE_integer("q_levels", 4, "the number of quantization levels in the output")

# training
flags.DEFINE_float("max_epoch", 100000, "maximum # of epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, color-mnist, cifar]")
flags.DEFINE_string("runtime_base_dir", "./", "path of base directory for checkpoints, data_dir, logs and sample_dir")
flags.DEFINE_string("data_dir", "data", "name of data directory")
flags.DEFINE_string("sample_dir", "samples", "name of sample directory")

# generation
flags.DEFINE_string("occlude_start_row", 18, "image row to start occlusion")
flags.DEFINE_string("num_generated_images", 9, "number of images to generate")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)


def validate_parameters(conf):
  if conf.data not in ["mnist", "color-mnist",  "cifar"]:
    raise ValueError("Configuration parameter 'data' is '{}'. Must be one of [mnist, color-mnist, cifar]"
                     .format(conf.data))


def preprocess(q_levels):
  
  def preprocess_fcn(images, labels):      
    # Create the target pixels from the image. Quantize the scalar pixel values into q_level indices.
    target_pixels = np.clip(((images * q_levels).astype('int64')), 0, q_levels - 1) # [N,H,W,C]
    return (images, target_pixels)
  
  return preprocess_fcn


def get_dataset(data_dir, q_levels):
  if conf.data == "mnist":  
    dataset = mnist.get_dataset(data_dir, preprocess(q_levels), reshape=False)
  elif conf.data == "color-mnist":
    dataset = mnist.get_colorized_dataset(data_dir, preprocess(q_levels), reshape=False)
  elif conf.data == "cifar":
    dataset = cifar.get_dataset(data_dir, preprocess(q_levels))

  return dataset


def generate_from_occluded(network, images):
  occlude_start_row = conf.occlude_start_row
  num_generated_images = conf.num_generated_images

  samples = network.generate_from_occluded(images, num_generated_images, occlude_start_row)
  
  occluded = np.copy(images[0:num_generated_images,:,:,:])
  # render white line in occlusion start row
  occluded[:,occlude_start_row,:,:] = 255
  return samples, occluded


def train(dataset, network, stat, sample_dir):
  initial_step = stat.get_t()
  logger.info("Training starts on epoch {}".format(initial_step))

  train_step_per_epoch = dataset.train.num_examples / conf.batch_size
  test_step_per_epoch = dataset.test.num_examples / conf.batch_size          

  for epoch in range(initial_step, conf.max_epoch):
    start_time = time.time()
    
    # 1. train
    total_train_costs = []        
    for _ in xrange(train_step_per_epoch):
      images = dataset.train.next_batch(conf.batch_size)
      cost = network.test(images, with_update=True)
      total_train_costs.append(cost)
    
    # 2. test        
    total_test_costs = []
    for _ in xrange(test_step_per_epoch):          
      images = dataset.test.next_batch(conf.batch_size)          
      cost = network.test(images, with_update=False)
      total_test_costs.append(cost)
      
    avg_train_cost, avg_test_cost = np.mean(total_train_costs), np.mean(total_test_costs)
    stat.on_step(avg_train_cost, avg_test_cost)
    
    # 3. generate samples
    images, _ = dataset.test.next_batch(conf.batch_size)
    samples, occluded = generate_from_occluded(network, images)
    util.save_images(np.concatenate((occluded, samples), axis=2), 
                dataset.height, dataset.width * 2, conf.num_generated_images, 1, 
                directory=sample_dir, prefix="epoch_%s" % epoch)
    
    logger.info("Epoch {}: {:.2f} seconds, avg train cost: {:.3f}, avg test cost: {:.3f}"
                .format(epoch,(time.time() - start_time), avg_train_cost, avg_test_cost))


def generate(network, height, width, sample_dir):
      logger.info("Image generation starts")
      samples = network.generate()
      util.save_images(samples, height, width, 10, 10, directory=sample_dir)


def main(_):
  model_dir = util.get_model_dir(conf, 
      ['data_dir', 'sample_dir', 'max_epoch', 'test_step', 'save_step',
       'is_train', 'random_seed', 'log_level', 'display', 'runtime_base_dir', 
       'occlude_start_row', 'num_generated_images'])
  util.preprocess_conf(conf)
  validate_parameters(conf)

  data = 'mnist' if conf.data == 'color-mnist' else conf.data 
  DATA_DIR = os.path.join(conf.runtime_base_dir, conf.data_dir, data)
  SAMPLE_DIR = os.path.join(conf.runtime_base_dir, conf.sample_dir, conf.data, model_dir)

  util.check_and_create_dir(DATA_DIR)
  util.check_and_create_dir(SAMPLE_DIR)
  
  dataset = get_dataset(DATA_DIR, conf.q_levels)

  with tf.Session() as sess:
    network = Network(sess, conf, dataset.height, dataset.width, dataset.channels)

    stat = Statistic(sess, conf.data, conf.runtime_base_dir, model_dir, tf.trainable_variables())
    stat.load_model()

    if conf.is_train:
      train(dataset, network, stat, SAMPLE_DIR)
    else:
      generate(network, dataset.height, dataset.width, SAMPLE_DIR)


if __name__ == "__main__":
  tf.app.run()
