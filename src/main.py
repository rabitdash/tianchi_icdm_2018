import numpy as np
import tools
from model.EFModel import EFModel
def main():
    m = EFModel('')
    # for raw_data in dataload.load_batch():
    #     pass
    # for i in range(1,6):
    #     data = np.load('../npydata/SRAD2018_TRAIN_010_Part{}_1000batch.npz'.format(i))['arr_0']
    #     m.train(data, size=10)
    # m.train(data)

    data = np.load('../npydata/SRAD2018_TRAIN_010_Part1_1000batch.npz')['arr_0']
    m.predict(data,steps=30)


if __name__ == '__main__':
    main()