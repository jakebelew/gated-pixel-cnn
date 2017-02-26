import mnist_data as mnist
import numpy as np
from data_test_utils import *


np.random.seed(39)
data_dir = "./"
height = 28
width = 28
train_label = 7

print("Testing MNIST dataset without preprocessing")
dataset = mnist.get_dataset(data_dir, reshape=False)
train_image = np.array([[ 0.44313725],
 [ 0.85882353],
 [ 0.99607843],
 [ 0.94901961],
 [ 0.89019608],
 [ 0.45098039],
 [ 0.34901961],
 [ 0.12156863],
 [ 0.        ],
 [ 0.        ],
 [ 0.        ],
 [ 0.        ],
 [ 0.78431373],
 [ 0.99607843],
 [ 0.94509804],
 [ 0.16078431]]
)
test_datasets(dataset, height, width, 1, train_image, train_label)

print("Testing Color-MNIST dataset without preprocessing")
dataset = mnist.get_colorized_dataset(data_dir, reshape=False)
train_image = np.array([[ 0., 0.44313729,  0.44313729],
 [ 0., 0.8588236,   0.8588236 ],
 [ 0., 0.99607849,  0.99607849],
 [ 0., 0.94901967,  0.94901967],
 [ 0., 0.89019614,  0.89019614],
 [ 0., 0.45098042,  0.45098042],
 [ 0., 0.34901962,  0.34901962],
 [ 0., 0.12156864,  0.12156864],
 [ 0., 0.,          0.        ],
 [ 0., 0.,          0.        ],
 [ 0., 0.,          0.        ],
 [ 0., 0.,          0.        ],
 [ 0., 0.7843138,   0.7843138 ],
 [ 0., 0.99607849,  0.99607849],
 [ 0., 0.9450981,   0.9450981 ],
 [ 0., 0.16078432,  0.16078432]]                       
)
test_datasets(dataset, height, width, 3, train_image, train_label)

q_levels = 4
print("Testing MNIST dataset with preprocessing")
dataset = mnist.get_dataset(data_dir, preprocess(q_levels), reshape=False)
train_image = np.array([[ 0.44313725],
 [ 0.85882353],
 [ 0.99607843],
 [ 0.94901961],
 [ 0.89019608],
 [ 0.45098039],
 [ 0.34901961],
 [ 0.12156863],
 [ 0.        ],
 [ 0.        ],
 [ 0.        ],
 [ 0.        ],
 [ 0.78431373],
 [ 0.99607843],
 [ 0.94509804],
 [ 0.16078431]]
)
train_label = np.array([[1],
 [3],
 [3],
 [3],
 [3],
 [1],
 [1],
 [0],
 [0],
 [0],
 [0],
 [0],
 [3],
 [3],
 [3],
 [0]]
)
test_datasets(dataset, height, width, 1, train_image, train_label)

print("Testing Color-MNIST dataset with preprocessing")
dataset = mnist.get_colorized_dataset(data_dir, preprocess(q_levels), reshape=False)
train_image = np.array([[ 0.44313725,  0.44313725,  0.44313725],
 [ 0.85882353,  0.85882353,  0.85882353],
 [ 0.99607843,  0.99607843,  0.99607843],
 [ 0.94901961,  0.94901961,  0.94901961],
 [ 0.89019608,  0.89019608,  0.89019608],
 [ 0.45098039,  0.45098039,  0.45098039],
 [ 0.34901961,  0.34901961,  0.34901961],
 [ 0.12156863,  0.12156863,  0.12156863],
 [ 0.,          0.,          0.        ],
 [ 0.,          0.,          0.        ],
 [ 0.,          0.,          0.        ],
 [ 0.,          0.,          0.        ],
 [ 0.78431373,  0.78431373,  0.78431373],
 [ 0.99607843,  0.99607843,  0.99607843],
 [ 0.94509804,  0.94509804,  0.94509804],
 [ 0.16078431,  0.16078431,  0.16078431]]
)
train_label = np.array([[1, 1, 1],
 [3, 3, 3],
 [3, 3, 3],
 [3, 3, 3],
 [3, 3, 3],
 [1, 1, 1],
 [1, 1, 1],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [3, 3, 3],
 [3, 3, 3],
 [3, 3, 3],
 [0, 0, 0]]
)
test_datasets(dataset, height, width, 3, train_image, train_label)

print("\n----- All tests passed -----")
