import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Concatenate, Multiply
from tensorflow.keras.layers import Flatten, Reshape, Dropout, Lambda
from tensorflow.keras.layers import Dense, Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.applications import resnet50

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
tf.disable_v2_behavior()

def mat_mul(A, B):
    return tf.matmul(A, B)

class PointTransNet(tf.keras.Model):
    def __init__(self, num_points):
        super(PointTransNet, self).__init__()
        self.num_points = num_points
        self.conv1 = Convolution1D(64, 1, activation='relu', input_shape=(self.num_points, 3))
        self.conv2 = Convolution1D(128, 1, activation='relu')
        self.conv3 = Convolution1D(1024, 1, activation='relu')
        self.max_pooling = MaxPooling1D(pool_size=self.num_points)
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(3 * 3, weights=[np.zeros([256, 3 * 3]), np.eye(3).flatten().astype(np.float32)])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pooling(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class FeatureTransNet(tf.keras.Model):
    def __init__(self, num_points):
        super(FeatureTransNet, self).__init__()
        self.num_points = num_points
        self.conv1 = Convolution1D(64, 1, activation='relu')
        self.conv2 = Convolution1D(128, 1, activation='relu')
        self.conv3 = Convolution1D(1024, 1, activation='relu')
        self.max_pooling = MaxPooling1D(pool_size=self.num_points)
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pooling(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Conv64(tf.keras.Model):
    def __init__(self, num_points):
        super(Conv64, self).__init__()
        self.num_points = num_points
        self.conv1 = Convolution1D(64, 1, input_shape=(self.num_points, 3), activation='relu')
        self.conv2 = Convolution1D(64, 1, input_shape=(self.num_points, 3), activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class Conv2048(tf.keras.Model):
    def __init__(self, num_points):
        super(Conv2048, self).__init__()
        self.num_points = num_points
        self.conv1 = Convolution1D(64, 1, activation='relu')
        self.conv2 = Convolution1D(128, 1, activation='relu')
        self.conv3 = Convolution1D(1024, 1, activation='relu')
        self.conv4 = Convolution1D(2048, 1, activation='relu')
        self.max_pooling = MaxPooling1D(pool_size=self.num_points)
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        return x

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(128, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Fusion(tf.keras.Model):
    def __init__(self, mode):
        super(Fusion, self).__init__()
        self.fusion = mode
        self.concat = Concatenate()
        self.add = Add()
        self.multiply = Multiply()

        self.dense = Dense(2048, activation='sigmoid')

    def call(self, inputs):
        if self.fusion == 'concatenate':
            x = self.concat(inputs)
        elif self.fusion == 'add':
            x = self.add(inputs)
        elif self.fusion == 'multiply':
            x = self.multiply(inputs)
        elif self.fusion == 'adaptive':
            c = self.concat(inputs)
            img_map = self.dense(c)
            img_feature = self.multiply([inputs[0], img_map])

            pc_map = self.dense(c)
            pc_feature = self.multiply([inputs[1], pc_map])
            x = self.add([img_feature, pc_feature])
        elif self.fusion == 'image':
            x = inputs[0]
        elif self.fusion == 'lidar':
            x = inputs[1]
        return x

