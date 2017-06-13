import numpy as np
import cv2

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, screen_height, screen_width = \
        config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.screen_height = screen_height
    self.screen_width = screen_width

    self.history = np.zeros([history_length, screen_height, screen_width], dtype=np.int32)

  def add(self, screen):
    # resize screen image size added by haeyong
    screen = np.array(screen, dtype=np.uint8)
    screen = cv2.resize(screen, (self.screen_height ,self.screen_width), interpolation=cv2.INTER_CUBIC)

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
