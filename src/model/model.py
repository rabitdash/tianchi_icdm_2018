import numpy as np
import pylab as plt
from tensorflow.contrib import keras
Sequential = keras.models.Sequential
Conv3D = keras.layers.Conv3D
ConvLSTM2D = keras.layers.ConvLSTM2D
BatchNormalization = keras.layers.BatchNormalization


# Constants
# 接受50x50图片作为输入
# 数组格式为[n_sample, n_frame, row, col, value]
IMAGE_SHAPE = {'width':50, 'height':50}
BATCH_SIZE = 10
DATA_PATH = '../data'


class Model(object):
    def __init__(self, weights_file='./weights.h5'):
        self.weights_file = weights_file
        self.seq = Sequential()
        self.history = {}
        self.seq.add(ConvLSTM2D(filters=40, 
                        kernel_size=(5, 5),
                        input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        return_sequences=True))
        self.seq.add(BatchNormalization())
        self.seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                        padding='same', return_sequences=True))
        self.seq.add(BatchNormalization())

        
        # self.seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
        #                 padding='same', return_sequences=True))
        # self.seq.add(BatchNormalization())

        # self.seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
        #                 padding='same', return_sequences=True))
        # self.seq.add(BatchNormalization())

        
        self.seq.add(Conv3D(filters=1, 
                    kernel_size=(3, 3, 3),
                    activation='sigmoid',
                    padding='same', 
                    data_format='channels_last'))
                    
        self.seq.compile(loss='binary_crossentropy', optimizer='adadelta')
        if weights_file != '':
            self.seq.load_weights(weights_file)

    def train(self, data, size=5, epoch=10, validation_split=0.05):
        for i in range(5):
            self.history = self.seq.fit(data[::,:-1,::,::,::], 
                        data[::,1:,::,::,::], 
                        batch_size=size,
                        epochs=epoch, 
                        validation_split=0.05)
            # print(self.history)
            self.seq.save_weights('./weights.h5')
            print("weights saved")
         
    def predict(self, data, steps=7):
        track2 = data[3][::,::,::,::]
        track = data[3][:steps,::,::,::]
        a = len(data[3])
        for j in range(a - steps + 1):
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
