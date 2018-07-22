
import tensorflow as tf
from convlstm_cell import ConvLSTMCell as LSTM
# from convlstm_cell import ConvGRUCell as GRU

#Constants
TRAIN_STEPS = 20000
IMAGE_SHAPE = [501, 501]
CHANNELS = 1 #image channels
BATCH_SIZE = 5
FILTER_SHAPE = [3, 3]
FEATURE_MAP_DEPTH = 32
DATA_PATH = '../data'
HIDDEN_SIZE = 200
NUM_LAYERS = 2
LEARNING_RATE = 1.0
NUM_STEPS = 61


inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS] + IMAGE_SHAPE + [channels])
lstm = LSTM(IMAGE_SHAPE, FEATURE_MAP_DEPTH, FILTER_SHAPE)
net = tf.nn.rnn_cell.MultiRNNCell([lstm] * NUM_LAYERS)
outputs, state = tf.nn.dynamic_rnn(net, inputs, dtype=inputs.dtype)

def loss():
   loss =