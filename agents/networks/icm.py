"""Neural networks that output value estimates for actions, given a state."""

import numpy as np
import tensorflow as tf
from pysc2.lib import features
from agents.preprocessing import preprocess_spatial_features
import os
from tensorflow.keras import initializers

SCREEN_FEATURES = features.SCREEN_FEATURES
NUM_ACTIONS = 1 
NUM_FEATURES = len(SCREEN_FEATURES)

# test (not sure if needed)
tf.compat.v1.disable_eager_execution()

class ICM(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None, 
                 name="ICM"
                 ):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path
        
        self.beta = 0.2  # changed from 0.2
        self.lamb = 1.0 # changed from 0.1
        self.eta = 0.01 # changed from 1.0 if you want r.i < 1, choose 0.0005, 0.05 gives good balance / 0.01

        # build graph
        self._build()
        
        # setup model saver
        if self.save_path:
            self.saver = tf.compat.v1.train.Saver(restore_sequentially=True)
            
    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def _build(self):
        """Construct graph."""
        with tf.compat.v1.variable_scope(self.name):
            # was int32
            self.state = tf.compat.v1.placeholder(
                tf.int32,
                [None, NUM_FEATURES, *self.spatial_dimensions],
                name="state")
            
            self.next_state = tf.compat.v1.placeholder(
                tf.int32,
                [None, NUM_FEATURES, *self.spatial_dimensions],
                name="next_state")
            
            self.screen_processed = preprocess_spatial_features(self.state, screen=True)
            self.next_screen_processed = preprocess_spatial_features(self.next_state, screen=True)

            with tf.compat.v1.variable_scope('conv'):
                self.s_conv1 = tf.compat.v1.layers.conv2d( 
                    inputs=self.screen_processed,
                    filters = 32, # changed from 32
                    kernel_size=[3, 3], 
                    strides=[2, 2], # changed 2,2
                    padding="SAME", 
                    activation=tf.nn.elu,
                    name="s_conv1")
                
                self.s_conv2 = tf.compat.v1.layers.conv2d(
                    inputs=self.s_conv1,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_conv2")
                
                self.s_conv3 = tf.compat.v1.layers.conv2d(
                    inputs=self.s_conv2,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_conv3")
                
                self.s_output = tf.compat.v1.layers.conv2d(
                    inputs=self.s_conv3,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_output")
                
                self.s_output_flat_dim = self.s_output.shape[1]*self.s_output.shape[2]*self.s_output.shape[3]
                self.s_flatten = tf.reshape(self.s_output, [tf.shape(self.s_output)[0], self.s_output_flat_dim], name = "s_flatten")
            
            with tf.compat.v1.variable_scope('conv', reuse=True):
                self.s_next_conv1 = tf.compat.v1.layers.conv2d(
                    inputs=self.next_screen_processed,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_conv1")
                
                self.s_next_conv2 = tf.compat.v1.layers.conv2d(
                    inputs=self.s_next_conv1,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_conv2")
                
                self.s_next_conv3 = tf.compat.v1.layers.conv2d(
                    inputs=self.s_next_conv2,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_conv3")
                
                self.s_next_output = tf.compat.v1.layers.conv2d(
                    inputs=self.s_next_conv3,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    activation=tf.nn.elu,
                    name="s_output")
                
                self.s_next_output_flat_dim = self.s_next_output.shape[1]*self.s_next_output.shape[2]*self.s_next_output.shape[3]
                self.s_next_flatten = tf.reshape(self.s_next_output, [tf.shape(self.s_next_output)[0], self.s_next_output_flat_dim], name = "s_next_flatten")
            
            # Forward model
            self.action = tf.compat.v1.placeholder(
                tf.float32,
                [None, np.prod((84, 84, NUM_ACTIONS))],
                name="action")

            
            self.input_forward = tf.concat([self.s_flatten, self.action], 1, name = "input_forward")
            
            self.dense_forward1 = tf.compat.v1.layers.dense(self.input_forward, 256, activation=tf.nn.relu,
                                                            name = "dense_forward1")
            self.dense_forward1 = tf.concat([self.dense_forward1, self.action], 1) # used to not be here remove this
            
            
            self.dense_forward2 = tf.compat.v1.layers.dense(self.dense_forward1, self.s_next_flatten.shape[1], activation=None,
                                                            name = "dense_forward2") # used to have activation relu
            
            self.loss_forward = tf.compat.v1.losses.mean_squared_error(self.dense_forward2, self.s_next_flatten)

            # Inverse model
            self.input_inverse = tf.concat([self.s_flatten, self.s_next_flatten], 1, name = "input_inverse")
            
            self.dense_inverse1 = tf.compat.v1.layers.dense(self.input_inverse, 256, activation=tf.nn.relu, # was 256
                                                            name = "dense_inverse1")
            # was np.prod((84, 84, NUM_ACTIONS))
            self.dense_inverse2 = tf.compat.v1.layers.dense(self.dense_inverse1, 
                                                            np.prod((84, 84, NUM_ACTIONS)),
                                                            activation=tf.nn.softmax,
                                                            name = "dense_inverse2")
            
            self.loss_inverse = tf.compat.v1.losses.softmax_cross_entropy(self.action, self.dense_inverse2)
            
            self.r_i = tf.multiply((self.eta * 0.5), tf.reduce_sum(tf.square(tf.subtract(self.dense_forward2, self.s_next_flatten)), axis = 1))
