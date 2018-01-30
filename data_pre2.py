# -*- coding: utf-8 -*-
"""
文件作用与其他说明:这是为lstm_keras.py 与 model_eva.py 准备数据
注意：1，更改分类个数要改#1
     2，这个函数是外部提取数据batch的接口
     def read_batchNPY(batch_size,file_dir)
        file_dir: your train root contain classNum ,like 0_dogs,1_cats
        return: a random [batchsize,[datashape]]
     3，这个函数是提取所以数据用于测试的接口
    def read_allTest(file_dir)
This is a file repare data for the lstm_keras and lstm_tesorflow

"""
import os
import random
import numpy as np
import threading
np.random.seed(7)
class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen
def onehot(index):
    """ It creates a one-hot vector with a 1.0 in
        position represented by index
    """
    classNum=2#1
    onehot = np.zeros(classNum)#这代表种类类型
    onehot[index] = 1.0
    return onehot
def read_batch(batch_size ,file_dir):
    """ It returns a batch of single npydata (no data-augmentation)
        Args:
            batch_size: need explanation? :)
            images_sources: path to training set folder
        Returns:
            batch_images: a tensor (numpy array of npydata) of shape [batch_size, timestep, feature_length]
            batch_labels: a tensor (numpy array of onehot vectors) of shape [batch_size, 1000]
    """
    batch_images = []
    batch_labels = []
    temp,size= get_files(file_dir)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    Size = size-1
    for i in range(batch_size):
        # random class choice
        # (randomly choose a folder of image of the same class from a list of previously sorted wnids)
        # class of the im
        class_index = random.randint(0, Size)
        batch_images.append(read_image(image_list[class_index]))
        batch_labels.append(onehot(int(label_list[class_index])))
    np.vstack(batch_images)
    np.vstack(batch_labels)
    return batch_images, batch_labels
def read_image(images_root):
    """ It road a single npy file into a numpy array and preprocess it
        Args:
            images_root: image path
        Returns:
            im_array: the numpy array of the npyarray [timestep, feature_length]
    """
    im_array = np.load(images_root)
    return im_array
def get_files(file_dir):
    '''
    Args:
        file_dir: file directory test or train
    Returns:
        list of data and labels
    '''
    trainImg = []
    label = []
    # 用于标志类别
    flag = 0
    for file in os.listdir(file_dir):
        # 有多少个文件夹有多少个类别，类别从0到结束作为标签
        # 最后flag返回输出的值就是有多少个类别
        # 为了更好规范图片数据，在分类的第一个字符写出是属于哪个类别
        flag = file[0]
        file = file_dir + file
        for img in os.listdir(file):
            trainImg.append(file + '/' + img)
            label.append(flag)
    flag = int(flag)
    flag += 1
    #flag可以代表种类数
    # print('There are %ds train data\n' % (len(trainImg)))
    # 把list转换成array,因为训练图片和label个数一样才能一列放一个
    temp = np.array([trainImg, label])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    size,_=np.shape(temp)
    return temp,size
def read_npy(npy_root):
    return np.load(npy_root)

###这两个函数是外部接口
@threadsafe_generator
def read_batchNPY(batch_size,file_dir):
    """
     batch_size: emmmm
     file_dir: your train root contain classNum ,like 0_dogs,1_cats
     return: a random [batchsize,[datashape]]
    """
    temp, size = get_files(file_dir)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    Size=size-1
    while 1:
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            # batch in seq
            class_index = random.randint(0, Size)
            batch_images.append(read_npy(image_list[class_index]))
            batch_labels.append(onehot(int(label_list[class_index])))
        np.vstack(batch_images)
        np.vstack(batch_labels)
        yield np.array(batch_images), np.array(batch_labels)

#生成所有数据用于测试
def read_allTest(file_dir):
    temp, size = get_files(file_dir)
    batch_images = []
    batch_labels = []
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    for i in range(size):
        # list all data in the test dir
        batch_images.append(read_image(image_list[i]))
        batch_labels.append(onehot(int(label_list[i])))
    np.vstack(batch_images)
    np.vstack(batch_labels)
    print ('总共有%d个文件用于测试',size)
    return np.array(batch_images), np.array(batch_labels)




