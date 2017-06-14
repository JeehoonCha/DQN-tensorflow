from __future__ import print_function
import os
import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config
from py4j.java_gateway import JavaGateway, GatewayParameters, Callbackserverpa

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'AngryBirdAI', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

MAX_STAGE = 21 + 1
MAX_ITER_PER_STAGE = 30
LEVEL_INFOS_FILENAME = 'level_infos.pickle'
MAX_TRAIN_ITER = 30
THREE_STAR = 'three_star'
TRAIN_ITER = 'train_step'
IS_TRAIN_CLEARED = 'is_train_cleared'
IS_PLAY_CLEARED = 'is_play_cleared'

import pickle
import os.path
def load_stage_infos():
  stage_infos = {}
  if os.path.isfile(LEVEL_INFOS_FILENAME):
    with open(LEVEL_INFOS_FILENAME, 'rb') as f:
      stage_infos = pickle.load(f)
  for level in range(1, MAX_STAGE):
    if level not in stage_infos:
      stage_infos[level] = {
        IS_TRAIN_CLEARED: False,
        IS_PLAY_CLEARED: False,
        THREE_STAR: False,
        TRAIN_ITER: 0,
      }
  return stage_infos

def save_stage_infos(stage_infos):
  with open(LEVEL_INFOS_FILENAME, 'wb') as f:
    pickle.dump(stage_infos, f)

def is_all_cleared(stage_infos):
  all_cleared = True
  for stage in stage_infos:
    if not stage_infos[stage][IS_PLAY_CLEARED]:
      all_cleared = False
  return all_cleared

import argparse
parser = argparse.ArgumentParser(description='argument parser for agent address')
parser.add_argument('--url', type=str, help='IP address to java agent client')
args = parser.parse_args()

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  config = get_config(FLAGS) or FLAGS
  gateway = JavaGateway(gateway_parameters=GatewayParameters(address=args.url)) if args.url else JavaGateway()
  actionRobot = gateway.entry_point

  if not tf.test.is_gpu_available() and FLAGS.use_gpu:
    raise Exception("use_gpu flag is true when no GPUs are available")
  if not FLAGS.use_gpu:
    config.cnn_format = 'NHWC'

  stage_infos = load_stage_infos()
  save_stage_infos(stage_infos)
  print (stage_infos)
  all_cleared = is_all_cleared(stage_infos)

  # Train until the agent clears all the levels
  if FLAGS.is_train:
    if not all_cleared:
      with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        agent = Agent(config, actionRobot, sess)

        for stage in stage_infos:
          agent.init_for_stage(stage)
          stage_infos[stage][IS_PLAY_CLEARED] = agent.play(stage, test_ep=0)
          if stage_infos[stage][IS_PLAY_CLEARED]:
            save_stage_infos(stage_infos)
          else:
            print ("Training agent... stage:" + str(stage))
            is_cleared = False
            while not (stage_infos[stage][IS_PLAY_CLEARED]):
              agent.init_for_stage(stage)
              train_iter = stage_infos[stage][TRAIN_ITER]
              is_train_cleared = agent.train_ep(stage, epsilon=1, train_iter=train_iter)
              stage_infos[stage][TRAIN_ITER] = train_iter + 1
              stage_infos[stage][IS_TRAIN_CLEARED] = is_train_cleared

              if (stage_infos[stage][IS_TRAIN_CLEARED]):
                stage_infos[stage][IS_PLAY_CLEARED] = agent.play(stage, test_ep=0)

              save_stage_infos(stage_infos)
              continue

  else:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      agent = Agent(config, actionRobot, sess)
      for stage in stage_infos:
        agent.init_for_stage(stage)
        stage_infos[stage][IS_PLAY_CLEARED] = agent.play(stage, test_ep=0)
        save_stage_infos(stage_infos)

if __name__ == '__main__':
  tf.app.run()
