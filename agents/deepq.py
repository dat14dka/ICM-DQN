"""Deep Q-learning agents."""
import numpy as np

"""If you wanna print entire matrices"""

import os
import tensorflow as tf
import functools
import math

# local submodule
import agents.networks.network as nets
import agents.networks.icm as icm

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS

# action space
attack_screen = functools.partial(FUNCTIONS.Attack_screen, "now")
action_space  = [attack_screen] 

NUM_ACTIONS   = len(action_space)

tf.executing_eagerly()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class DQNAgent(base_agent.BaseAgent):
    """A DQN that receives screen observations and selects units, moves and/or attacks."""

    def __init__(self,
                 learning_rate= FLAGS.learning_rate,
                 discount_factor=FLAGS.discount_factor,
                 epsilon_max=FLAGS.epsilon_max,
                 epsilon_min=FLAGS.epsilon_min,
                 epsilon_decay_steps=FLAGS.epsilon_decay_steps,
                 train_frequency=FLAGS.train_frequency,
                 target_update_frequency=FLAGS.target_update_frequency,
                 max_memory=FLAGS.max_memory,
                 batch_size=FLAGS.batch_size,
                 training=FLAGS.training,
                 indicate_nonrandom_action=FLAGS.indicate_nonrandom_action,
                 save_dir="./checkpoints/dqn/",
                 save_dir2= "./checkpoints/icm/", 
                 save_dir3= "./checkpoints/deepq+icm/",
                 ckpt_name = "DQNAgent",
                 ckpt_name2 = "ICM",
                 ckpt_name3 = "DeepQ+ICM",
                 graph_path = None, # set './tensorboard/graphs' if you want graphs
                 summary_path="./tensorboard/deepq_statistics"
                 ):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNAgent, self).__init__()

        # saving and summary writing
        if FLAGS.save_dir:
            save_dir = FLAGS.save_dir
        if FLAGS.ckpt_name:
            ckpt_name = FLAGS.ckpt_name
        if FLAGS.summary_path:
            summary_path = FLAGS.summary_path
        self.save_every_nth_episode = FLAGS.save_every_nth_episode

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.extrinsic_coeff = 1.0
        self.intrinsic_coeff = 0.01 

        # agent hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # other parameters
        self.training = training
        self.indicate_nonrandom_action = indicate_nonrandom_action
        self.adjust_coordinates = True
        self.avoid_self_move = True
        
        # these coordinates are determined by acient deepmind magic, test manually
        self.lower_bound_cut_off_x = 7
        self.upper_bound_cut_off_x = 76
        self.lower_bound_cut_off_y = 10
        self.upper_bound_cut_off_y = 55

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        self.save_path2 = save_dir2 + ckpt_name2 + ".ckpt"
        self.save_path3 = save_dir3 + ckpt_name3 + ".ckpt"
        self.graph_path = graph_path
        
        #test #verkar inte hjälpa, frågan är om det skadar?
        #tf.keras.backend.clear_session()
        print("Building models...")
        tf.compat.v1.reset_default_graph() # tf.reset_default_graph()
        self.network = nets.Network(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.Network(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                name="DQNTarget")

            # initialize Experience Replay memory buffer
            self.memory = Memory(max_memory)
            self.batch_size = batch_size

        # there seem,s to be an issue with loading checkpointse, possibly due to renaming variables,
        # or maybe due to the introduction of ICM. 
        # Might have to reset the dafault graph again here, but the error occurs in the self.network line above.
        # so this is only part of the solution at most...
        self.ICM = icm.ICM(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            summary_path=summary_path,
            save_path = self.save_path2, 
            name="ICM")
        
        self.loss_and_train()
        
        # setup model saver
        if self.save_path:
            self.saver = tf.compat.v1.train.Saver(restore_sequentially=True)
        
        # Create printer for graphs
        if graph_path:
            self.graph_writer = tf.compat.v1.summary.FileWriter(graph_path, tf.compat.v1.get_default_graph())
        
        print("Done.")

        self.last_state = None
        self.last_action = None

        # initialize session
        self.sess = tf.compat.v1.Session(config=config) #config=config

                
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
            self.ICM.load(self.sess)
            self.load(self.sess)
            if self.training:
                self._update_target_network()
        else:
                self._tf_init_op() #tab

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.score = 0
        self.total_loss = 0.0

        if self.training:
            self.last_state = None
            self.last_action = None

            global_episode = self.network.global_episode.eval(session=self.sess)

            print("Global training episode:", global_episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.score += obs.reward 
        
        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        state = obs.observation.feature_screen
        
        if self.training:
            # predict an action to take and take it
            action, action_index = self._epsilon_greedy_action_selection(state)
            
            # update online DQN
            if (self.steps % self.train_frequency == 0 and
                    len(self.memory) > self.batch_size):
                _ = self._train_network() 
                    
            # update network used to estimate TD targets
            if self.steps % self.target_update_frequency == 0:
                self._update_target_network()
                print("Target network updated.")                

            # add experience to memory
            if self.last_state is not None:
                self.memory.add(
                    (self.last_state,
                     self.last_action,
                     self.score,
                     state))

            self.last_state = state
            self.last_action = action_index
            
        else:
            action, _ = self._epsilon_greedy_action_selection(
                state,
                self.epsilon_min) #TODO should be 0

        action_name = action.function.name

        if action_name == 'Attack_screen':
            action_id = 12
        else:
            action_id = 0
        return action if action_id in obs.observation.available_actions else FUNCTIONS.no_op()

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # increment global training episode
        self.network.increment_global_episode_op(self.sess)
        global_episode = self.network.global_episode.eval(session=self.sess)

        if(global_episode % self.save_every_nth_episode == 0):
            self.network.save_model(self.sess)
            print("DeepQ Model Saved")
            
            self.ICM.save_model(self.sess)
            print("ICM Model Saved")
            
            self.save_model(self.sess)
            print("ICM+DQN Saved")
        
        # write summaries from last episode
        states, actions, _, _, _, _, targets = self._get_batch()
            
        self.network.write_summary(
            self.sess, states, actions, targets, 
            self.score, self.total_loss)
                
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op, options=run_opts)

    def _update_target_network(self):
        online_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        target_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNTarget")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op, options=run_opts)
        
    def _epsilon_greedy_action_selection(self, state, epsilon=None):
        """Choose action from state with epsilon greedy strategy."""   
        
        episode = self.network.global_episode.eval(session=self.sess)
        if (epsilon is None):
            denominator=(episode*0.003)
            epsilon = math.exp(-denominator)

        if (epsilon > np.random.rand()):
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])
            action_index = np.random.randint(0, len(action_space))
            action = action_space[action_index]
            max_index = np.ravel_multi_index((x, y, action_index), (84, 84, NUM_ACTIONS))
        else:
            inputs = np.expand_dims(state, 0)

            q_values = self.network.output.eval(feed_dict={self.network.inputs: inputs}, session = self.sess) 
            
            max_index = np.argmax(q_values)
            x, y, action_index = np.unravel_index(max_index, (84, 84, NUM_ACTIONS))
            action = action_space[action_index]
            
        _TERRAN_MARINE = 48
        unit_type = state.unit_type
        cc_y, cc_x = (unit_type == _TERRAN_MARINE).nonzero()
        result = zip(cc_x, cc_y)
        result = set(result)
        
        if(not self.coordinatesAreLegitimate(x, y, result, action.func.name)):
            return FUNCTIONS.no_op(), max_index
          
        #return action(result[test]), max_index     
        #x,y = self.coordinatesAreLegitimate(x, y, result, action.func.name)         
        return action((x, y)), max_index


    # should return any intrinsic rewards
    def _train_network(self):
        states, actions, next_states, _, _, r_i, targets = self._get_batch()
        
        #loss
        loss, _, _, _, _, _ = self.sess.run( #do I need r_i here ?
            [self.loss, self.loss_DQN , self.loss_INV, self.loss_FOR, self.ICM.r_i, self.train_step],
            feed_dict={self.ICM.state: states,
                       self.ICM.action: actions, #was action
                       self.ICM.next_state: next_states,
                       self.network.inputs: states,
                       self.network.actions: actions,
                       self.network.targets: targets
                       }, options=run_opts) 

        self.total_loss = loss
        
        """ This prints any weights you might want ! """
           
        return -1 # just placeholder
        
    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
                

        # one-hot encode actions
        actions = np.eye(np.prod((84, 84, NUM_ACTIONS)))[actions]
                
        # get targets
        next_outputs = self.sess.run(
            self.target_net.output,
            feed_dict={self.target_net.inputs: next_states})

        
        r_i = self.ICM.r_i.eval(session = self.sess, feed_dict = {self.ICM.state: states, 
                                                   self.ICM.next_state: next_states, 
                                                   self.ICM.action: actions})
                
        for i in range(len(rewards)):                
            rewards[i] = (self.extrinsic_coeff * rewards[i]) + (self.intrinsic_coeff * r_i[i])
        
        targets = []
        for i in range(len(rewards)):
            targets.append(rewards[i] + self.discount_factor * np.max(next_outputs[i]))


        return states, actions, next_states, next_outputs, rewards, r_i, targets
    
    
    def loss_and_train(self):
		# Loss function and train
        self.loss_DQN = tf.multiply(self.ICM.lamb, self.network.loss)
        self.loss_FOR = tf.multiply(self.ICM.beta, self.ICM.loss_forward)
        self.loss_INV = tf.multiply((1-self.ICM.beta), self.ICM.loss_inverse)
        self.loss = tf.add_n([self.loss_DQN, self.loss_INV, self.loss_FOR])

        self.train_step = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.network.global_step)
        
    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path3)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path3)
        
    def coordinatesAreLegitimate(self, x, y, coordinates, action_name):        
        if(self.adjust_coordinates and (action_name == "Move_screen" or action_name == "Attack_screen")):
            if((x < self.lower_bound_cut_off_x) or (x > self.upper_bound_cut_off_x)):
                return False
            if((y < self.lower_bound_cut_off_y) or (y > self.upper_bound_cut_off_y)):
                return False
        
        for coordinate in coordinates:
            if(self.inBufferZone(x,y,coordinate)):
                return False;
              
        #if(self.avoid_self_move and ((x,y) in coordinates)):
        #        return False
         
        return True
    
    def inBufferZone(self, x, y, coordinate):
        bufferSize = 3
        if((x > (coordinate[0]-bufferSize) and x < (coordinate[0]+bufferSize)) and 
           (y > (coordinate[1]-bufferSize) and y < (coordinate[1]+bufferSize))):
                return True
        return False;
            
            
        
        
    
    
