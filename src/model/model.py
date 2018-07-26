from tensorflow.contrib import keras
Sequential = keras.models.Sequential
Conv3D = keras.layers.Conv3D
ConvLSTM2D = keras.layers.ConvLSTM2D
BatchNormalization = keras.layers.BatchNormalization
import numpy as np
import pylab as plt

#Constants
IMAGE_SHAPE = {'width':50, 'height':50}
BATCH_SIZE = 10
DATA_PATH = '../data'
filter_size = 50
stddev=0.1
class Model():
    def __init__(self, weights_file='./weights.h5'):
        self.train_count = 0
        self.weights_file = weights_file
        self.seq = Sequential()
        self.history = {}
        # self.seq.add(keras.layers.InputLayer(
        #     input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
        #     dtype=np.float32)
        # )
        # self.seq.add(BatchNormalization())
        # self.seq.add(keras.layers.Conv2D(filters=1, kernel_size=(3,3), data_format='channels_last'))
        # self.seq.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        # self.seq.add(keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), data_format='channels_last'))
        # self.seq.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.seq.add(ConvLSTM2D(filters=20, 
                        kernel_size=(3, 3),
                        input_shape=(None, IMAGE_SHAPE['width'], IMAGE_SHAPE['height'], 1),
                        padding='same', 
                        return_sequences=True))
        # self.seq.add(keras.layers.MaxPooling3D(pool_size=(5,5,5),padding='same'))
        self.seq.add(BatchNormalization())
        # self.seq.add(keras.layers.MaxPooling3D(2,2,2))
        # self.seq.add(keras.layers.GaussianNoise(stddev))
        self.seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                        padding='same', return_sequences=True))
        # self.seq.add(keras.layers.MaxPooling3D((5,5,5), padding='same'))
        self.seq.add(BatchNormalization())

        
        self.seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                        padding='same', return_sequences=True))
        # self.seq.add(keras.layers.MaxPooling3D((2,2,2), padding='same'))
        self.seq.add(BatchNormalization())

        self.seq.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                        padding='same', return_sequences=True))
        # self.seq.add(keras.layers.UpSampling3D((5,5,5)))
        self.seq.add(BatchNormalization())

        
        self.seq.add(Conv3D(filters=1, 
                    kernel_size=(3, 3, 3),
                    activation='tanh',
                    padding='same', 
                    data_format='channels_last'))
        # self.seq.add(keras.layers.LeakyReLU(alpha=0.3))
                    
        self.seq.compile(loss='binary_crossentropy', optimizer='rmsprop')
        if weights_file != '':
            self.seq.load_weights(weights_file)

    # def set_train_data(self, data):
    #     self.data = np.array(data)

    def train(self, data, size=8, epoch=10, validation_split=0.05):
        # print("Train step{}".format(self.train_count))
        # data = data[::,::,::,::,::]
        # data = data[::,::,:250,:250,::]
        for i in range(0,10):
            self.history = self.seq.fit(data[::,:-1,::,::,::], 
                        data[::,1:,::,::,::], 
                        batch_size=size,
                        epochs=epoch, 
                        validation_split=0.05)
            
            self.seq.save_weights('./weights.h5')
            print("weights saved")
         
    def predict(self, data, steps=7):
        # data = data[::,::,::,::,::]
        # data = data[::,::,:50,:50,::]
        track2 = data[4][::,::,::,::]
        track = data[4][:steps,::,::,::]
        a = len(data[4])
        for j in range(a - steps + 1):
            new_pos = self.seq.predict(track[np.newaxis, ::, ::, ::,::])
            new = new_pos[::, -1, ::, ::,::]
            track = np.concatenate((track, new), axis=0)

        for i in range(a):
            fig = plt.figure(figsize=(10, 5))

            ax = fig.add_subplot(121)

            if i >= steps:
                ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
            else:
                ax.text(1, 3, 'Initial trajectory', fontsize=20)

            toplot = track[i, ::, ::, 0]

            plt.imshow(toplot)
            ax = fig.add_subplot(122)
            plt.text(1, 3, 'Ground truth', fontsize=20)

            toplot = track2[i, ::, ::, 0]
            # if i >= 2:
            #     toplot = data[3][i - 1, ::, ::, 0]
            plt.imshow(toplot)
            plt.savefig('../prediction/%i_animate.png' % (i + 1))
