import numpy as np
import model.model
import dataload
def main():
    m = model.model.Model('')
    # for raw_data in dataload.load_batch():
    #     pass
    for i in range(1,6):
        data = np.load('../npydata/SRAD2018_TRAIN_010_Part{}_1000batch.npz'.format(i))['arr_0']
    # data = raw_data[::,::,::,::,::]
    # # data = data / 255.0
        m.train(data)
    # data = raw_data[::,::,::,::,::]
    # data = data / 255.0
    # m.train(data)
    # data = raw_data[::,::,::5,::5,::]
    # data = data/255.0
    m.predict(data, steps=30)

if __name__ == '__main__':
    main()