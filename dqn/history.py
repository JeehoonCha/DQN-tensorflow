import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, state_length, state_width = \
        config.batch_size, config.history_length, config.state_length, config.state_width

    self.history = np.zeros(
        [history_length, state_length, state_width], dtype=np.int32)

  def add(self, screen_bytes):
    screen = self.convertToNumpyArray(screen_bytes, self.history[-1].shape)
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen
    print "screen added to history"

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history

  def convertToNumpyArray(self, screen_bytes, shape):
    return np.frombuffer(screen_bytes, dtype='>u4').reshape(shape)
