from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
from .ops import linear, conv2d, clipped_error
from .utils import get_time, save_pkl, load_pkl

VALUE_TO_NORMALIZE = 100
MAX_ANGLE = 90
MIN_INTERVAL = 60
MAX_INTERVAL = 90
MAX_NUM_BIRDS = 8
SCREEN_Y_FROM = 200
SCREEN_Y_UNTIL = 400
SCREEN_X_FROM = 420
SCREEN_X_UNTIL = 750

class Agent(BaseModel):
  def __init__(self, config, actionRobot, sess, stage):
    print ("config=" + str(config))
    self.sess = sess
    self.weight_dir = 'weights'
    self.stage = stage
    self.agent = actionRobot

    actionRobot.loadLevel(stage)
    self.action_size = MAX_ANGLE
    self.interval_size = MAX_INTERVAL - MIN_INTERVAL
    config.max_step = actionRobot.getMaxStep()
    super(Agent, self).__init__(config)

    self.screen_shape = (config.screen_height, config.screen_width)
    self.origin_screen_shape = (480, 840)
    self.action_screen_shape = (SCREEN_Y_UNTIL - SCREEN_Y_FROM, SCREEN_X_UNTIL - SCREEN_X_FROM)
    self.history = History(self.config)

    self.history.add(actionRobot.getScreen())
    self.memory = ReplayMemory(self.config, self.model_dir)
    self.maximum_reward = 18000

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.random_normal_mean = 0
    self.random_normal_sigma = 0.005

    self.q_action_value = 0
    self.n_action_value = 0

    print ("Building Deep Q Network..")
    self.build_dqn(self.stage)
    self.load_trained_network()

  def train_ep(self, stage, epsilon=None, train_iter=None):
    # initialization
    self.update_count = 0
    self.total_loss, self.total_q = 0., 0.

    self.stage = stage
    start_step = 1

    # agent load the specific stage
    observation = self.agent.loadLevel(self.stage)

    screen = self.convert_screen_to_numpy_array(observation.getScreen())
    reward = observation.getReward()
    terminal = observation.getTerminal()
    self.history.add(screen)

    print('stage:', stage, 'start_step:', start_step, 'state:', self.agent.getGameState())
    for self.step in tqdm(range(start_step, MAX_NUM_BIRDS), ncols=70, initial=start_step):
      # 1. predict
      self.step_input = self.step

      # 2. act
      angle = self.predict(self.history.get(), test_ep=epsilon, train_iter=train_iter) # Pick action based on Q-Network
      tabInterval = 200 # TODO(jeehoon): need to modify
      print('angle:', angle, 'tabInterval:', tabInterval)

      observation = self.agent.shoot(int(angle), tabInterval)
      screen = self.convert_screen_to_numpy_array(observation.getScreen())
      reward = observation.getReward()
      terminal = observation.getTerminal()

      reward_score = np.abs(reward)
      reward_ratio = min(int(np.abs(reward)), self.maximum_reward) / float(self.maximum_reward)
      if reward_score == 0: # The bird hits nothing
        reward_ratio = -1.0
      elif str(self.agent.getGameState()) == 'WON':
        reward_ratio = 1.0

      print('step:',self.step,
            'reward_score:', reward_score,
            'norm_reward:', reward_ratio,
            'state:', self.agent.getGameState())

      # 3. observe
      self.observe(screen, reward_ratio, angle, terminal)

      state = self.agent.getGameState()
      if str(state) in ['WON', 'LOST'] or reward > 20000 : # The stage is finished while there still some birds left.
        break

    if epsilon > 0:
      self.save_trained_network()

    return str(state) == 'WON'

  def predict(self, s_t, test_ep=None, train_iter=None):
    print ("config.train_max_iter:",self.config.train_max_iter,
           "self.train_max_iter:",self.train_max_iter,
           "self.ep_start:", str(self.ep_start),
           "test_ep:", str(test_ep),
           "train_iter:", str(train_iter))
    ep = test_ep or self.ep_end + \
                    max(0., (self.ep_start - self.ep_end) * (self.train_max_iter - train_iter) / self.train_max_iter) # ep: prob. to pick random action

    ep_rnd = random.random()
    if ep_rnd < ep:
      shootInfo = self.agent.getShootInfo() # JavaShootingAgent gives angle and power info
      angle = shootInfo.getAngle()
      angle += np.random.normal(self.random_normal_mean, self.random_normal_sigma) * 100 # normally distributed value
      angle = min(angle, MAX_ANGLE) # angle cannot exceed MAX_ANGLE
      action = angle
    else:
      q_action = self.q_action.eval({self.s_t: [s_t]})[0]
      self.q_action_value = q_action
      action = q_action

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.target_q_update_step = 1
    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)
    self.q_learning_mini_batch()
    self.update_target_q_network()

  def q_learning_mini_batch(self):
    s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
    t = time.time()
    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      })
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    else:
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
    })

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def build_dqn(self, stage):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1') # ?, 32, 119, 209
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2') # ?, 64,58, 103
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3') # ? 64, 56, 101

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])]) # ?, 361984

      if self.dueling:
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage -
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4') # ?, 512
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=1) # ?, 10000

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0) # 10000
      for idx in xrange(self.action_size):
        q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1') # ? 32, 119, 209
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2') # ? 64, 58, 103
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3') # ? 64, 56, 101

      shape = self.target_l3.get_shape().as_list() # ? , 361984
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage -
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4') # ?, 512
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.action_size, name='target_q') # ?, 10000

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted

      # Euclidean distance between x1,x2
      self.delta_action = tf.reduce_mean(tf.square(self.delta))

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
      self.loss_action = tf.reduce_mean(tf.square(self.delta_action), name='loss_action')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

      #self.optim_action_d = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss_action)

      self.optim_action = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss_action)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        print ("tag=" + str(tag) + ", env_name=" + str(self.env_name))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    # self.load_model(stage)
    # self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_trained_network(self):
    self.save_model(self.stage)
    self.save_weight_to_pkl()

  def load_trained_network(self):
    self.load_model(self.stage)
    self.load_weight_from_pkl()

  def save_weight_to_pkl(self):
    dir_path = os.path.join(self.weight_dir, str(self.stage))
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(dir_path, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    dir_path = os.path.join(self.weight_dir, str(self.stage))
    if not os.path.exists(dir_path):
      return

    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(dir_path, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, stage, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end
    self.train_ep(stage, test_ep)

  def convert_screen_to_numpy_array(self, screen_bytes):
    return np.frombuffer(screen_bytes, dtype='>u4').reshape(self.origin_screen_shape)[
           SCREEN_Y_FROM:SCREEN_Y_UNTIL,
           SCREEN_X_FROM:SCREEN_X_UNTIL]