#coding: utf-8
import os
import numpy as np
import imageio
import multiprocessing
import time
import zipfile
filename_definition = "{}_Part{}" #{folder_definition:}
batch_size = 1000
save_path = '../../npydata/'
n_build_process = 1 # 多进程数
n_frames = 61
# data format: np.Array(n_samples, n_frames, row, col, channels)
# channels default is 1, n_frames=61
def build(path='../../data'):
    folders = os.listdir(path)
    folders.sort()
    
    #去除重复文件夹
    # folders = list(set(folders + built_folder))

    for folder in folders:
        print('start build {}'.format(folder))
        samples = ['/'.join([path,folder, x]) for x in os.listdir('/'.join([path, folder]))]
        # samples.sort()
        n_part = 0
        p = multiprocessing.Pool(n_build_process)
        for i in range(0, len(samples), batch_size):
            n_part += 1
            arg_samples_path = samples[i:i+batch_size]
            p.apply_async(build_samples, args=(n_part, arg_samples_path, folder))
        p.close()
        p.join()
        print('finish build {}'.format(folder))

def build_samples(n_part, samples_path, folder_name):
    movie = np.zeros((batch_size, 61, 51, 51, 1), dtype=np.float)
    build_time = time.time()
    print('start build {} part {}'.format(folder_name, n_part))
    # sample_list = []
        # save
    save_filename = save_path  \
                    + filename_definition.format(folder_name,n_part) \
                    + "_{}batch".format(batch_size) \
                    + ".npz"
    if os.path.exists(save_filename):
        print('previously finished build {} part {}'.format(folder_name, n_part))
        return
    for i, sample in zip(range(len(samples_path)), samples_path):
        pics = os.listdir('/'.join([sample]))
        # pics_list = []
        pics.sort()
        for j, pic in zip(range(len(pics)), pics):
            pic_nparray = np.array(imageio.imread('/'.join([sample, pic])),
                                    dtype=np.float32)
            movie[i,j,::,::,0] = pic_nparray[::10,::10] # downsample
            print(pic)
        print("part{} progress {}/{}".format(n_part, i + 1, batch_size))
    movie = movie[::,::,:50,:50,::] # cut picture
    print(len(movie[0][0]))
    movie = movie / 255.0 # normalize
    compress_time = time.time()
    print("part{} start compress".format(n_part))
    np.savez_compressed(save_filename,
        movie)
    compress_time = time.time() - compress_time
    print("part{} finish compress, time:{}".format(n_part, compress_time))
    build_time = time.time() - build_time
    print('finished build {} part {}, time:{}'.format(folder_name, n_part, build_time))

def load_data(path='../../npydata'):
    data = os.listdir()
    return np.load('/'.join([path, data[0]]), dtype=float)

def downsample(data, scale):
    return data[::,::,::scale,::scale,::]

def cut_frame(data, split):
    pass
if __name__ == '__main__':
    build()
