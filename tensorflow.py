import os
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import gzip
from mnist import MNIST

root_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

seed = 128
rng = np.random.RandomState(seed)

mndata = MNIST(data_dir)

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

print(len(X_train[0]))