import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras.api.keras.layers import TimeDistributed, Input ,InputLayer
from .Layers import ConvLSTM2D
from .Layers import ConvGRU2D
Sequential = keras.models.Sequential
Conv3D = keras.layers.Conv3D
IMAGE_SHAPE={'width':501, 'height':501}

BatchNormalization = keras.layers.BatchNormalization
EncoderInputs = Input(shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1), name='EncoderInputs')
ForecasterInputs = Input(shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1), name='ForecasterInputs')
#共享RNN层
EncoderRNN1 = ConvGRU2D(filters=5, kernel_size=(5,5),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        return_sequences=True,
                        strides=(5,5),
                        )
EncoderRNN2 = ConvGRU2D(filters=20, kernel_size=(3,3),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        strides=(5,5),
                        return_sequences=True)
EncoderRNN3 = ConvGRU2D(filters=40, kernel_size=(3,3),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        strides=(2,2),
                        return_sequences=True)
ForecasterRNN3 = ConvGRU2D(filters=40, kernel_size=(3,3),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        strides=(2,2),
                        return_sequences=True)
ForecasterRNN2 = ConvGRU2D(filters=20, kernel_size=(3,3),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        strides=(5,5),
                        return_sequences=True)
ForecasterRNN1 = ConvGRU2D(filters=5, kernel_size=(3,3),
                        # input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        strides=(5,5),
                        return_sequences=True)                        
# Convolution层
ConvIn = TimeDistributed(keras.layers.Convolution2D(filters=1, kernel_size=(3,3), padding='same', data_format='channels_last'))
ConvOut = TimeDistributed(keras.layers.Convolution2D(filters=1, kernel_size=(3,3), padding='same', data_format='channels_last'))

# ConvIn = Conv3D(filters=1, 
#                 kernel_size=(3, 3, 3),
#                 activation='sigmoid',
#                 padding='same', 
#                 data_format='channels_last')
Conv5 = TimeDistributed(keras.layers.Convolution2D(filters=5, kernel_size=(1,1), padding='same', data_format='channels_last'))
Conv10 = TimeDistributed(keras.layers.Convolution2D(filters=10, kernel_size=(1,1), padding='same', data_format='channels_last'))
ConvMiddle1 = Conv3D(filters=1, 
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', 
                data_format='channels_last')
ConvMiddle2 = Conv3D(filters=1, 
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', 
                data_format='channels_last')
ConvMiddle3 = Conv3D(filters=1, 
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', 
                data_format='channels_last')
# DownSample层
DownSample1 = TimeDistributed(keras.layers.MaxPool2D(pool_size=(1,1),padding='same', data_format='channels_last'))
DownSample2 = TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2),padding='same', data_format='channels_last'))
DownSample3 = TimeDistributed(keras.layers.MaxPool2D(pool_size=(5,5),padding='same', data_format='channels_last'))
# UpSample层
UpSample1 = TimeDistributed(keras.layers.UpSampling2D(size=(5,5), data_format='channels_last'))
UpSample2 = TimeDistributed(keras.layers.UpSampling2D(size=(2,2), data_format='channels_last'))
# UpSample3 = TimeDistributed(keras.layers.UpSampling2D(size=(1,1), data_format='channels_last'))
# Ouputs
Outputs = Conv3D(filters=1, 
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', 
                data_format='channels_last')
# Constants
# 接受50x50图片作为输入
# 数组格式为[n_sample, n_frame, row, col, value]
DATA_PATH = '../data'

def generate_movies(n_samples=120, n_frames=61):
    row = 400
    col = 400
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)
        for j in range(n):
            # Initial position
            xstart = np.random.randint(200, 600)
            ystart = np.random.randint(200, 600)
            # Direction of motion
            directionx = np.random.randint(0, 30) - 10
            directiony = np.random.randint(0, 30) - 10

            # Size of the square
            w = np.random.randint(20, 40)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 400x400 window
    # noisy_movies = noisy_movies[::, ::, 200:600, 200:600, ::]
    # shifted_movies = shifted_movies[::, ::, 200:600, 200:600, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies	
class EFModel(object):
    def __init__(self, weights_file='./weights.h5'):
        self.weights_file = weights_file
        self.seq = Sequential()
        # self.seq.add(InputLayer(input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1)))
        # self.seq.add(ConvIn)

        # self.seq.add(DownSample1)
        # self.seq.add(EncoderRNN1)
        # self.seq.add(BatchNormalization())

        # self.seq.add(DownSample2)
        # self.seq.add(EncoderRNN2)
        # self.seq.add(BatchNormalization())

        # # self.seq.add(DownSample3)
        # # self.seq.add(EncoderRNN3)
        # # self.seq.add(BatchNormalization())

        # # self.seq.add(UpSample1)
        # # self.seq.add(EncoderRNN2)
        # # self.seq.add(BatchNormalization())

        # self.seq.add(UpSample2)
        # self.seq.add(EncoderRNN1)
        # self.seq.add(BatchNormalization())

        # # self.seq.add(UpSample3)
        # # self.seq.add(BatchNormalization())
        # self.seq.add(Outputs)


        # self.seq.add(ConvOut)
        # predictions = ConvOut(Forecaster)
        # # Inputs = ConvIn(Inputs)
        # Encoder
        Encoder = ConvIn(EncoderInputs)

        Encoder, last_state1 = EncoderRNN1(Encoder)
        # last_state1 = ConvMiddle1(Encoder)[-1]
        Encoder = ConvMiddle1(Encoder)

        Encoder, last_state2 = EncoderRNN2(Encoder)
        # last_state2 = ConvMiddle2(Encoder)[-1]
        Encoder = ConvMiddle2(Encoder)
        Encoder = DownSample2(Encoder)

        Encoder, last_state3 = EncoderRNN3(Encoder)
        # Encoder = ConvOut(Encoder)
        # last_state3 = ConvMiddle3(Encoder)[-1]

        # Forecaster = EncoderRNN3(last_state3)
        # print(last_state3.get_shape())
        Forecaster = ForecasterRNN3(ForecasterInputs, initial_state=last_state3)
        # Forecaster = ConvMiddle2(Forecaster)
        # Forecaster = UpSample2(Forecaster)
        
        # Forecaster = ForecasterRNN2(Forecaster, initial_state=last_state2)
        # Forecaster = ConvMiddle1(Forecaster)
        # Forecaster = UpSample1(Forecaster)

        # Forecaster = ForecasterRNN1(Forecaster, initial_state=last_state1)
        predictions = ConvOut(Encoder) 

        self.model = keras.models.Model(inputs=[EncoderInputs], outputs=predictions)
        self.model.compile(loss='binary_crossentropy', optimizer='adadelta')
                    
        if weights_file != '':
            self.seq.load_weights(weights_file)
            print('Load weights')
        else:
            self.weights_file='./weights.h5'
        # self.noisy_movies, self.shifted_movies = generate_movies(n_samples=50)
    def train(self, data, size=1, epoch=10, validation_split=0.05):
        """
            Forecaster_Inputs[::,t,::,::,::] = Forecaster_Outputs[::,t+1,::,::,::]
        """
        for i in range(10):
            # self.model.fit(data, 
            #             data, 
            #             batch_size=size,
            #             epochs=epoch, 
            #             validation_split=0.05)

            # self.model.fit([Encoder_Inputs, Forecaster_Inputs],Forecasters_Outputs)
            self.model.fit(data[::,:31,::,::,::], data[::,31:,::,::,::], batch_size=size, epochs=epoch, validation_split=0.05)
            self.model.save_weights(self.weights_file)
            print("weights saved")
         
    def predict(self, data, steps=7):
        # data = self.shifted_movies
        track2 = data[2][::,::,::,::]
        track = data[2][:steps,::,::,::]
        a = len(data[2])
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
