'''
Modules used in the project.
'''
import os

from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, add, Activation, AvgPool2D
from keras.layers import BatchNormalization
from keras.layers import concatenate, GaussianNoise
from keras.models import Sequential, Model
from keras.preprocessing.image import *
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

import numpy as np

from fun import *