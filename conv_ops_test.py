import numpy as np
import tensorflow as tf

from ops import conv

def run(op):
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    outputs = sess.run(op)    
    print("outputs shape: {}:".format(outputs.shape))
    print(np.transpose(outputs, axes=(0,3,1,2))[0,:,:,:])
    return outputs

def create_inputs(image_shape, num_inputs):
  num_batch = 1
  image_h, image_w = image_shape
  inputs_shape = [num_batch, image_h, image_w, num_inputs]
  return np.arange(1, np.prod(inputs_shape) + 1, dtype=np.float32).reshape(inputs_shape)

def create_ones_inputs(image_shape, num_inputs):
  num_batch = 1
  image_h, image_w = image_shape
  inputs_shape = [num_batch, image_h, image_w, num_inputs]
  return tf.ones(inputs_shape, dtype=tf.float32)
  
def expected_from_list(e_list): 
  e_list = np.array(e_list)
  e_list = np.transpose(e_list, axes=(1,2,0))
  e_list = np.expand_dims(e_list, axis=0) 
  return e_list

def assert_equals(array, expected):
  if not np.array_equal(array, expected):
    raise Exception("Array is not what is expected")

def matrix_to_string(array): 
  '''Creates a string that is in the proper format to create a numpy array from.'''
  np.set_printoptions(precision=0)
  string = '['
  h, w = array.shape
  for i in range(h):
    if not (i == 0):
      string += ' '
    string += '['
    for j in range(w):
      string += ' {0:.0f}'.format(array[i,j]) + '.'
      if (j < w-1):
        string += ', '            
    string += ']'
    if (i < h-1):
      string += ',\n'      
  return string + ']'
  
def test_first_layer():
  print("===========================================")
  print("First Layer: 7 x 7 conv mask A (1 layer)")
  print("Grayscale [1 in, 16 out]")  
  input_shape = [9,9]
  kernel_shape = [7,7]
  mask_type = "A"
  num_inputs, num_outputs, data_num_channels = 1, 2, 1

  expected_0 = np.array(
  [[ 0.,  1.,  2.,  3.,  3.,  3.,  3.,  3.,  3.],
   [ 4.,  6.,  8.,  10.,  10.,  10.,  9.,  8.,  7.],
   [ 8.,  11.,  14.,  17.,  17.,  17.,  15.,  13.,  11.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.],
   [ 12.,  16.,  20.,  24.,  24.,  24.,  21.,  18.,  15.]])
  expected = expected_from_list([expected_0, expected_0])

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="7x7_conv_mask_A_Grayscale"))
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  
  print("-------------------------------------------")
  print("Color [3 in, 48 out]")
  num_inputs, num_outputs, data_num_channels = 3, 6, 3

  expected_0 = np.array(
  [[ 0.,  3.,  6.,  9.,  9.,  9.,  9.,  9.,  9.],
   [ 12.,  18.,  24.,  30.,  30.,  30.,  27.,  24.,  21.],
   [ 24.,  33.,  42.,  51.,  51.,  51.,  45.,  39.,  33.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.],
   [ 36.,  48.,  60.,  72.,  72.,  72.,  63.,  54.,  45.]])
  
  expected_1 = np.array(
  [[ 1.,  4.,  7.,  10.,  10.,  10.,  10.,  10.,  10.],
   [ 13.,  19.,  25.,  31.,  31.,  31.,  28.,  25.,  22.],
   [ 25.,  34.,  43.,  52.,  52.,  52.,  46.,  40.,  34.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.],
   [ 37.,  49.,  61.,  73.,  73.,  73.,  64.,  55.,  46.]])
  
  expected_2 = np.array(
  [[ 2.,  5.,  8.,  11.,  11.,  11.,  11.,  11.,  11.],
   [ 14.,  20.,  26.,  32.,  32.,  32.,  29.,  26.,  23.],
   [ 26.,  35.,  44.,  53.,  53.,  53.,  47.,  41.,  35.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.],
   [ 38.,  50.,  62.,  74.,  74.,  74.,  65.,  56.,  47.]])    
  
  expected = expected_from_list([expected_0, expected_1, expected_2] * 2)

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="7x7_conv_mask_A_Color"))
  #print(matrix_to_string(outputs[0,:,:,2]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  

def test_gated_conv_3x3_layer():
  print("===========================================")
  print("Gated Conv Layers: 3 x 3 gated conv mask B (Multiple layers)")
  print("Grayscale [16 in, 16 out]")
  input_shape = [5,5]
  kernel_shape = [3,3]
  mask_type = "B"
  num_inputs, num_outputs, data_num_channels = 2, 2, 1
  
  expected_0 = np.array(
  [[ 2.,  4.,  4.,  4.,  4.],
   [ 6.,  10.,  10.,  10.,  8.],
   [ 6.,  10.,  10.,  10.,  8.],
   [ 6.,  10.,  10.,  10.,  8.],
   [ 6.,  10.,  10.,  10.,  8.]])
   
  expected = expected_from_list([expected_0, expected_0])

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="3x3_gated_conv_mask_B_Grayscale"))  
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  
  print("-------------------------------------------")
  print("Color [3 in, 48 out]")
  num_inputs, num_outputs, data_num_channels = 3, 6, 3

  expected_0 = np.array(
  [[ 1.,  4.,  4.,  4.,  4.],
   [ 7.,  13.,  13.,  13.,  10.],
   [ 7.,  13.,  13.,  13.,  10.],
   [ 7.,  13.,  13.,  13.,  10.],
   [ 7.,  13.,  13.,  13.,  10.]])
    
  expected_1 = np.array(
  [[ 2.,  5.,  5.,  5.,  5.],
   [ 8.,  14.,  14.,  14.,  11.],
   [ 8.,  14.,  14.,  14.,  11.],
   [ 8.,  14.,  14.,  14.,  11.],
   [ 8.,  14.,  14.,  14.,  11.]])
    
  expected_2 = np.array(
  [[ 3.,  6.,  6.,  6.,  6.],
   [ 9.,  15.,  15.,  15.,  12.],
   [ 9.,  15.,  15.,  15.,  12.],
   [ 9.,  15.,  15.,  15.,  12.],
   [ 9.,  15.,  15.,  15.,  12.]])  
  
  expected = expected_from_list([expected_0, expected_1, expected_2] * 2)

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="3x3_gated_conv_mask_B_Color"))  
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  

def test_gated_horiz_mask_layer():
  print("===========================================")
  print("Gated Conv Layers: 1 x 3 gated horiz conv mask B (Multiple layers)")
  print("Grayscale [16 in, 16 out]")
  input_shape = [5,5]
  kernel_shape = [1,3]
  mask_type = "B"
  num_inputs, num_outputs, data_num_channels = 2, 2, 1
  
  expected_0 = np.array(
  [[ 2.,  4.,  4.,  4.,  4.],
   [ 2.,  4.,  4.,  4.,  4.],
   [ 2.,  4.,  4.,  4.,  4.],
   [ 2.,  4.,  4.,  4.,  4.],
   [ 2.,  4.,  4.,  4.,  4.]])
   
  expected = expected_from_list([expected_0, expected_0])

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="1x3_gated_horiz_conv_mask_B_Grayscale"))  
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  
  print("-------------------------------------------")
  print("Color [3 in, 48 out]")
  num_inputs, num_outputs, data_num_channels = 3, 6, 3

  expected_0 = np.array(
  [[ 1.,  4.,  4.,  4.,  4.],
   [ 1.,  4.,  4.,  4.,  4.],
   [ 1.,  4.,  4.,  4.,  4.],
   [ 1.,  4.,  4.,  4.,  4.],
   [ 1.,  4.,  4.,  4.,  4.]])
    
  expected_1 = np.array(
  [[ 2.,  5.,  5.,  5.,  5.],
   [ 2.,  5.,  5.,  5.,  5.],
   [ 2.,  5.,  5.,  5.,  5.],
   [ 2.,  5.,  5.,  5.,  5.],
   [ 2.,  5.,  5.,  5.,  5.]])
    
  expected_2 = np.array(
  [[ 3.,  6.,  6.,  6.,  6.],
   [ 3.,  6.,  6.,  6.,  6.],
   [ 3.,  6.,  6.,  6.,  6.],
   [ 3.,  6.,  6.,  6.,  6.],
   [ 3.,  6.,  6.,  6.,  6.]])  
                        
  expected = expected_from_list([expected_0, expected_1, expected_2] * 2)

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="1x3_gated_horiz_conv_mask_B_Color"))  
  
  #print(matrix_to_string(outputs[0,:,:,2]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  

def test_last_layers():
  print("===========================================")
  print("Last Layers: 1 x 1 conv mask B (2 layers)")
  print("Grayscale - (No masking required) [32 in, 32 out]")
  input_shape = [5,5]
  kernel_shape = [1,1]
  mask_type = "B"
  num_inputs, num_outputs, data_num_channels = 2, 2, 1
  
  expected_0 = np.array(
  [[ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.]])
                           
  expected = expected_from_list([expected_0, expected_0])

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="1x1_conv_mask_B_Grayscale"))    
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  
  print("-------------------------------------------")
  print("Color [3 in, 48 out]")
  num_inputs, num_outputs, data_num_channels = 3, 6, 3

  expected_0 = np.array(
  [[ 1.,  1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.,  1.]])
      
  expected_1 = np.array(
  [[ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.],
   [ 2.,  2.,  2.,  2.,  2.]])
      
  expected_2 = np.array(
  [[ 3.,  3.,  3.,  3.,  3.],
   [ 3.,  3.,  3.,  3.,  3.],
   [ 3.,  3.,  3.,  3.,  3.],
   [ 3.,  3.,  3.,  3.,  3.],
   [ 3.,  3.,  3.,  3.,  3.]])
                         
  expected = expected_from_list([expected_0, expected_1, expected_2] * 2)

  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, mask_type, data_num_channels, weights_initializer=tf.ones_initializer, scope="1x1_conv_mask_B_Color"))      
  #print(matrix_to_string(outputs[0,:,:,2]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  

def repeat_matrix(pattern, num_rows, num_columns):
  pattern = np.repeat(pattern[:, :, np.newaxis], num_rows, axis=2)
  return np.repeat(pattern[:, :, :, np.newaxis], num_columns, axis=3)

def test_masked_vert_mask_layer():
  print("===========================================")
  print("Gated Conv Layers: 1 x 3 vertical conv mask (Multiple layers)")
  print("Grayscale [16 in, 16 out]")
  input_shape = [5,5]
  kernel_shape = [3,3]
  num_inputs, num_outputs, data_num_channels = 2, 2, 1
  
  expected = np.array(
  [[ 0.,  0.,  0.,  0.,  0.],
   [ 4.,  6.,  6.,  6.,  4.],
   [ 4.,  6.,  6.,  6.,  4.],
   [ 4.,  6.,  6.,  6.,  4.],
   [ 4.,  6.,  6.,  6.,  4.]])
     
  expected = expected_from_list([expected] * num_outputs)
  
  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, 'V', data_num_channels, weights_initializer=tf.ones_initializer, scope="1x3_vert_conv_mask_Grayscale"))      
  #print(matrix_to_string(outputs[0,:,:,0]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  
  print("-------------------------------------------")
  print("Color [3 in, 48 out]")
  num_inputs, num_outputs, data_num_channels = 3, 6, 3

  expected = np.array(
  [[ 0.,  0.,  0.,  0.,  0.],
   [ 6.,  9.,  9.,  9.,  6.],
   [ 6.,  9.,  9.,  9.,  6.],
   [ 6.,  9.,  9.,  9.,  6.],
   [ 6.,  9.,  9.,  9.,  6.]])
     
  expected = expected_from_list([expected] * num_outputs)
  
  outputs = run(conv(
    create_ones_inputs(input_shape, num_inputs), 
    num_outputs, kernel_shape, 'V', data_num_channels, weights_initializer=tf.ones_initializer, scope="1x3_vert_conv_mask_Color"))        
  #print(matrix_to_string(outputs[0,:,:,2]))
  # invalid to crop_mask for A or B, only horiz or vertical mask can crop

  assert_equals(outputs, expected)  


np.set_printoptions(suppress=True)  

test_first_layer()
test_gated_conv_3x3_layer()
test_gated_horiz_mask_layer()
test_last_layers()
test_masked_vert_mask_layer()

print("\n***** All tests passed *****")
