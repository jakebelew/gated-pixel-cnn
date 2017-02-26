from logging import getLogger

import tensorflow as tf

from ops import *
from utils import *

logger = getLogger(__name__)

class Network:

  def __init__(self, sess, conf, height, width, num_channels):
    logger.info("Building gated_pixel_cnn starts")

    self.sess = sess
    self.data = conf.data
    self.height, self.width, self.channel = height, width, num_channels
    self.pixel_depth = 256
    self.q_levels = q_levels = conf.q_levels

    self.inputs = tf.placeholder(tf.float32, [None, height, width, num_channels]) # [N,H,W,C]
    self.target_pixels = tf.placeholder(tf.int64, [None, height, width, num_channels]) # [N,H,W,C] (the index of a one-hot representation of D)

    # input conv layer
    logger.info("Building CONV_IN")    
    net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], "A", num_channels, scope="CONV_IN")
    
    # main gated layers
    for idx in xrange(conf.gated_conv_num_layers):
      scope = 'GATED_CONV%d' % idx
      net = gated_conv(net, [3, 3], num_channels, scope=scope)
      logger.info("Building %s" % scope)

    # output conv layers
    net = tf.nn.relu(conv(net, conf.output_conv_num_feature_maps, [1, 1], "B", num_channels, scope='CONV_OUT0'))
    logger.info("Building CONV_OUT0")
    self.logits = tf.nn.relu(conv(net, q_levels * num_channels, [1, 1], "B", num_channels, scope='CONV_OUT1')) # shape [N,H,W,DC]
    logger.info("Building CONV_OUT1")
      
    if (num_channels > 1):
      self.logits = tf.reshape(self.logits, [-1, height, width, q_levels, num_channels]) # shape [N,H,W,DC] -> [N,H,W,D,C]            
      self.logits = tf.transpose(self.logits, perm=[0, 1, 2, 4, 3]) # shape [N,H,W,D,C] -> [N,H,W,C,D]             
    
    flattened_logits = tf.reshape(self.logits, [-1, q_levels]) # [N,H,W,C,D] -> [NHWC,D] 
    target_pixels_loss = tf.reshape(self.target_pixels, [-1]) # [N,H,W,C] -> [NHWC]
    
    logger.info("Building loss and optims")    
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
       flattened_logits, target_pixels_loss))

    flattened_output = tf.nn.softmax(flattened_logits) #shape [NHWC,D], values [probability distribution]
    self.output = tf.reshape(flattened_output, [-1, height, width, num_channels, q_levels]) #shape [N,H,W,C,D], values [probability distribution]

    optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)

    new_grads_and_vars = \
        [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
    self.optim = optimizer.apply_gradients(new_grads_and_vars)
 
    show_all_variables()

    logger.info("Building gated_pixel_cnn finished")

  def predict(self, images):
    '''
    images # shape [N,H,W,C]
    returns predicted image # shape [N,H,W,C]
    '''
    # self.output shape [NHWC,D]
    pixel_value_probabilities = self.sess.run(self.output, {self.inputs: images}) # shape [N,H,W,C,D], values [probability distribution]
    
    # argmax or random draw # [NHWC,1]  quantized index - convert back to pixel value    
    pixel_value_indices = np.argmax(pixel_value_probabilities, 4) # shape [N,H,W,C], values [index of most likely pixel value]
    pixel_values = np.multiply(pixel_value_indices, ((self.pixel_depth - 1) / (self.q_levels - 1))) #shape [N,H,W,C]

    return pixel_values

  def test(self, images, with_update=False):
    if with_update:
      _, cost = self.sess.run([self.optim, self.loss], 
                              { self.inputs: images[0], self.target_pixels: images[1] })
    else:
      cost = self.sess.run(self.loss, { self.inputs: images[0], self.target_pixels: images[1] })
    return cost

  def generate_from_occluded(self, images, num_generated_images, occlude_start_row):
    samples = np.copy(images[0:num_generated_images,:,:,:])
    samples[:,occlude_start_row:,:,:] = 0.

    for i in xrange(occlude_start_row,self.height):
      for j in xrange(self.width):
        for k in xrange(self.channel):
          next_sample = self.predict(samples) / (self.pixel_depth - 1.) # argmax or random draw here
          samples[:, i, j, k] = next_sample[:, i, j, k]
    
    return samples

  def generate(self, images):
    samples = images[0:9,:,:,:]
    occlude_start_row = 18
    samples[:,occlude_start_row:,:,:] = 0.
    
    for i in xrange(occlude_start_row,self.height):
      for j in xrange(self.width):
        for k in xrange(self.channel):
          next_sample = self.predict(samples) / (self.pixel_depth - 1.) # argmax or random draw here
          samples[:, i, j, k] = next_sample[:, i, j, k]
    
    return samples
