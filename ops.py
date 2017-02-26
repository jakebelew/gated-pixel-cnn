import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

logger = logging.getLogger(__name__)

def get_shape(layer):
  return layer.get_shape().as_list()

def conv(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, 'A', 'B' or 'V'
    data_num_channels,
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="conv2d"):
  with tf.variable_scope(scope):
    mask_type = mask_type.lower()
    if mask_type == 'v' and kernel_shape == [1,1]:
      # No mask required for Vertical 1x1 convolution
      mask_type = None
    num_inputs = get_shape(inputs)[-1]

    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      "kernel height and width should be an odd number"

    weights_shape = [kernel_h, kernel_w, num_inputs, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    if mask_type is not None:
      mask = _create_mask(num_inputs, num_outputs, kernel_shape, data_num_channels, mask_type)
      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    logger.debug('[conv2d_%s] %s : %s %s -> %s %s' \
        % (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

# for this type layer: num_outputs = num_inputs = number of channels in input layer 
def gated_conv(inputs, kernel_shape, data_num_channels, scope="gated_conv"):
  with tf.variable_scope(scope):
    # Horiz inputs/outputs on left in case num_inputs not multiple of 6, because Horiz is RGB gated and Vert is not.
    # inputs shape [N,H,W,C]
    horiz_inputs, vert_inputs  = tf.split(3, 2, inputs)  
    p = get_shape(horiz_inputs)[-1]
    p2 = 2 * p

    # vertical n x n conv
    # p in channels, 2p out channels, vertical mask, same padding, stride 1
    vert_nxn = conv(vert_inputs, p2, kernel_shape, 'V', data_num_channels, scope="vertical_nxn")
      
    # vertical blue diamond
    # 2p in channels, p out channels, vertical mask
    vert_gated_out = _gated_activation_unit(vert_nxn, kernel_shape, 'V', data_num_channels, scope="vertical_gated_activation_unit")
    
    # vertical 1 x 1 conv
    # 2p in channels, 2p out channels, no mask?, same padding, stride 1
    vert_1x1 = conv(vert_nxn, p2, [1, 1], 'V', data_num_channels, scope="vertical_1x1")

    # horizontal 1 x n conv
    # p in channels, 2p out channels, horizontal mask B, same padding, stride 1
    horiz_1xn = conv(horiz_inputs, p2, kernel_shape, 'B', data_num_channels, scope="horizontal_1xn")
    horiz_gated_in = vert_1x1 + horiz_1xn
    
    # horizontal blue diamond
    # 2p in channels, p out channels, horizontal mask B
    horiz_gated_out = _gated_activation_unit(horiz_gated_in, kernel_shape, 'B', data_num_channels, scope="horizontal_gated_activation_unit")    
    
    # horizontal 1 x 1 conv
    # p in channels, p out channels, mask B, same padding, stride 1
    horiz_1x1 = conv(horiz_gated_out, p, kernel_shape, 'B', data_num_channels, scope="horizontal_1x1")
    
    horiz_outputs = horiz_1x1 + horiz_inputs
     
    return tf.concat(3, [horiz_outputs, vert_gated_out]) 
  
def _create_mask(    
    num_inputs,
    num_outputs,
    kernel_shape,
    data_num_channels,
    mask_type, # 'A', 'B' or 'V'
    ):
    '''
    Produces a causal mask of the given type and shape
    '''
    mask_type = mask_type.lower()
    kernel_h, kernel_w = kernel_shape
    
    center_h = kernel_h // 2
    center_w = kernel_w // 2

    mask = np.ones(
      (kernel_h, kernel_w, num_inputs, num_outputs), dtype=np.float32) # shape [KERNEL_H, KERNEL_W, NUM_INPUTS, NUM_OUTPUTS]
    
    if mask_type == 'v':
      mask[center_h:, :, :, :] = 0.
    else:
      mask[center_h, center_w+1:, :, :] = 0.
      mask[center_h+1:, :, :, :] = 0.
      
      if mask_type == 'b':
        mask_pixel = lambda i,j: i > j
      else:
        mask_pixel = lambda i,j: i >= j
        
      for i in range(num_inputs):
        for j in range(num_outputs):
          if mask_pixel(i % data_num_channels, j % data_num_channels):
            mask[center_h, center_w, i, j] = 0.

    return mask
     
# implements equation (2) of the paper
# returns 1/2 number of channels as input
def _gated_activation_unit(inputs, kernel_shape, mask_type, data_num_channels, scope="gated_activation_unit"):
  with tf.variable_scope(scope):
    p2 =  get_shape(inputs)[-1]    

    # blue diamond
    # 2p in channels, 2p out channels, mask, same padding, stride 1
    # split 2p out channels into p going to tanh and p going to sigmoid
    bd_out = conv(inputs, p2, kernel_shape, mask_type, data_num_channels, scope="blue_diamond") #[N,H,W,C[,D]]
    bd_out_0, bd_out_1 = tf.split(3, 2, bd_out)
    tanh_out = tf.tanh(bd_out_0)
    sigmoid_out = tf.sigmoid(bd_out_1)
      
  return tanh_out * sigmoid_out