"""Neural networks that output value estimates for actions, given a state."""

import numpy as np
import tensorflow as tf
from pysc2.lib import features
from agents.preprocessing import preprocess_spatial_features

SCREEN_FEATURES = features.SCREEN_FEATURES
NUM_ACTIONS = 1
NUM_FEATURES = len(SCREEN_FEATURES)

# test (not sure if needed)
tf.compat.v1.disable_eager_execution()
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)


class Network(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="DQN"):
        
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            #print("++++++++++ entered network")
            self.writer = tf.compat.v1.summary.FileWriter(summary_path)
            tf.compat.v1.summary.scalar("Loss_per_episode", self.Total_loss)
            tf.compat.v1.summary.scalar("Batch_Max_Q_per_episode", self.max_q)
            tf.compat.v1.summary.scalar("Batch_Mean_Q_per_episode", self.mean_q)
            tf.compat.v1.summary.scalar("Score_per_episode", self.score)
            self.write_op = tf.compat.v1.summary.merge_all()
            
            if isinstance(self.write_op, type(None)):
                # Will crash after 1 episode if True 
                print(" +-+ +-+ +-+ self.write_op is none! +-+ +-+ +-+")

        # setup model saver
        if self.save_path:
            self.saver = tf.compat.v1.train.Saver(restore_sequentially=True)

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        print("------------network sp:", self.save_path)
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score, 
                      total_loss):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.Total_loss: total_loss,
                       self.score: score,
                       }, options=run_opts)
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode, options=run_opts)

    def _build(self):
        """Construct graph."""
        with tf.compat.v1.variable_scope(self.name):
        
            # trackers
            self.score = tf.compat.v1.placeholder(
                tf.float32,
                [],
                name="score")
            
            self.Total_loss = tf.compat.v1.placeholder(
                tf.float32,
                
                [],
                name="Total_loss")

            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")
                
            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")
    
            # network architecture
            self.inputs = tf.compat.v1.placeholder(
                tf.int32,
                [None, NUM_FEATURES, *self.spatial_dimensions],
                name="inputs")

            self.increment_global_episode = tf.compat.v1.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            self.screen_processed = preprocess_spatial_features(self.inputs, screen=True)

            # tf.keras.layers.conv2d?
            self.conv1 = tf.compat.v1.layers.conv2d(
                inputs=self.screen_processed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="SAME",
                activation=tf.nn.relu,
                name="conv1")

            # tf.keras.layers.conv2d?
            self.conv2 = tf.compat.v1.layers.conv2d(
                inputs=self.conv1,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                activation=tf.nn.relu,
                name="conv2")

            # tf.keras.layers.conv2d?
            self.output = tf.compat.v1.layers.conv2d(
                inputs=self.conv2,
                filters=NUM_ACTIONS,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="output")

            # tf.keras.layers?
            self.flatten = tf.compat.v1.layers.flatten(self.output, name="flat")
            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flatten, name="max")

            self.mean_q = tf.reduce_mean(self.flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.actions = tf.compat.v1.placeholder(
                tf.float32,
                [None, np.prod((84, 84, NUM_ACTIONS))],
                name="actions")

            self.targets = tf.compat.v1.placeholder(
                tf.float32,
                [None],
                name="targets")

            self.prediction = tf.reduce_sum(
                tf.multiply(self.flatten, self.actions),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")