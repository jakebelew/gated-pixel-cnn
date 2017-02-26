import numpy as np


def test_dataset(dataset, image=None, label=None):
  images = dataset.images  
  labels = dataset.labels
  if image is not None:
    print(images[0,18,6:22,:])
    np.testing.assert_almost_equal(images[0,18,6:22,:], image)
  if label is not None:
    if isinstance(label, int):
      print(labels[0])
      np.testing.assert_almost_equal(labels[0], label)
    else:
      print(labels[0,18,6:22,:])
      np.testing.assert_almost_equal(labels[0,18,6:22,:], label)
          

def test_datasets(dataset, height, width, channels, train_image, train_label):
  assert(dataset.height == height)
  assert(dataset.width == width)
  assert(dataset.channels == channels)
  test_dataset(dataset.train, train_image, train_label)


def preprocess(q_levels):
  
  def preprocess_fcn(images, labels):      
    # Create the target pixels from the image. Quantize the scalar pixel values into q_level indices.
    target_pixels = np.clip(((images * q_levels).astype('int64')), 0, q_levels - 1) # [N,H,W,C]
    return (images, target_pixels)
  
  return preprocess_fcn
