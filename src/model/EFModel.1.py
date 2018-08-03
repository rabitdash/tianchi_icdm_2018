import numpy as np
import pylab as plt
from tensorflow.contrib import keras
from tensorflow.contrib.keras.api.keras.layers import TimeDistributed, InputLayer
from .Layers import ConvLSTM2D
from .Layers import ConvGRU2D
Sequential = keras.models.Sequential
Conv3D = keras.layers.Conv3D
Functional = keras.models.Model()
IMAGE_SHAPE={'width':50, 'height':50}
# ConvLSTM2D = keras.layers.ConvLSTM2D
BatchNormalization = keras.layers.BatchNormalization
Inputs = InputLayer(shape=(None, 61, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1))
#共享RNN层
SharedRNN1 = ConvGRU2D(filters=20, kernel_size=(5, 5),
                        input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', return_sequences=True)
SharedRNN2 = ConvGRU2D(filters=20, kernel_size=(5, 5),
                        input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', return_sequences=True)
SharedRNN3 = ConvGRU2D(filters=20, kernel_size=(5, 5),
                        input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', return_sequences=True)
# DownSample层
DownSample = TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2),padding='same', data_format='channel_last'))
# UpSample层
UpSample = TimeDistributed(keras.layers.UpSampling2D(size=(2,2), data_format='channel_last'))
# Constants
# 接受50x50图片作为输入
# 数组格式为[n_sample, n_frame, row, col, value]
DATA_PATH = '../data'


class EFModel(object):
    def __init__(self, weights_file='./weights.h5'):
        self.weights_file = weights_file
        # self.seq = Sequential()
        Encoder = Inputs
        Encoder = DownSample(Encoder)
        Encoder = SharedRNN1(Encoder)
        Encoder = DownSample(Encoder)
        Encoder = SharedRNN2(Encoder)
        Encoder = DownSample(Encoder)
        Encoder = SharedRNN3(Encoder)
        # self.history = {}
        # self.seq.add(ConvLSTM2D(filters=20, 
        #                 kernel_size=(5, 5),
        #                 input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
        #                 padding='same', 
        #                 return_sequences=True))
        # self.seq.add(BatchNormalization())
        # self.seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
        #                 padding='same', return_sequences=True))
        # self.seq.add(BatchNormalization())

        # self.seq.add(keras.layers.MaxPooling2D())
        # self.seq.add(ConvGRU2D(filters=20, kernel_size=(5, 5),
        #                 input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
        #                 padding='same', return_sequences=True))
        # self.seq.add(BatchNormalization())

        # self.seq.add(ConvGRU2D(filters=40, kernel_size=(3, 3),
        #                 padding='same', return_sequences=True))
        # self.seq.add(BatchNormalization())

        
        # self.seq.add(Conv3D(filters=1, 
        #             kernel_size=(3, 3, 3),
        #             activation='sigmoid',
        #             padding='same', 
        #             data_format='channels_last'))
                    
        # self.seq.compile(loss='binary_crossentropy', optimizer='adadelta')
        # if weights_file != '':
        #     self.seq.load_weights(weights_file)
        # else:
        #     self.weights_file='./weights.h5'

    def train(self, data, size=5, epoch=1, validation_split=0.05):
        for i in range(50):
            # self.history = self.seq.fit(data[::,:-1,::,::,::], 
            #             data[::,1:,::,::,::], 
            #             batch_size=size,
            #             epochs=epoch, 
            #             validation_split=0.05)
            # self.seq.save_weights(self.weights_file)
            print("weights saved")
         
    def predict(self, data, steps=7):
        track2 = data[3][::,::,::,::]
        track = data[3][:steps,::,::,::]
        a = len(data[3])
        for j in range(a - steps):
            new_pos = self.seq.predict(track[np.newaxis, ::, ::, ::,::])
            new = new_pos[::, -1, ::, ::,::]
            track = np.concatenate((track, new), axis=0)

        for i in range(a):
            fig = plt.figure(figsize=(10, 5))

            ax = fig.add_subplot(121)

            if i >= steps:
                ax.text(1, 3, 'Predictions', fontsize=20, color='w')
            else:
                ax.text(1, 3, 'Initial trajectory', fontsize=20)

            toplot = track[i, ::, ::, 0]

            plt.imshow(toplot)
            ax = fig.add_subplot(122)
            plt.text(1, 3, 'Ground truth', fontsize=20)

            toplot = track2[i, ::, ::, 0]
            plt.imshow(toplot)
            plt.savefig('../prediction/%02d_animate.png' % (i + 1))
