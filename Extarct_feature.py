# -*- coding: utf-8 -*-
#文件作用与其他说明:用于提取训练好的caffemode的特征，参考于caffe官方例子00-classification.ipynb
#1，每个文件夹内100张图片，为一个孩子一次癫痫发作，每张图片为两秒。命名方式为a（1）到a（100）
#2，排序后按照顺序提取每张图片的1000维特征存放为100*1000的矩阵 npy格式

import numpy as np
import sys
import os
caffe_root = '/home/a504/nvcaffe/caffe'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/a504/nvcaffe/caffe/python')
import caffe
net_file = '/home/a504/PycharmProjects/caffe_lstm/test3/deploy.prototxt'#1
caffe_model = '/home/a504/PycharmProjects/caffe_lstm/test3/snapshot_iter_64000.caffemodel' #2
mean_file ='/home/a504/PycharmProjects/caffe_lstm/test3/mean.binaryproto'#3


print('Params loaded!')

caffe.set_mode_cpu()

net = caffe.Net(net_file,
                caffe_model,
                caffe.TEST)
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_file, 'rb').read())
mean_npy = caffe.io.blobproto_to_array(mean_blob)
a = mean_npy[0, :, 0, 0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', a)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))

#图片存放与存储路径 在 caffe CNN + keras Lstm readme.txt中有相关的截图示例
typeDirPath='/home/a504/data/new_starting/test_nm'        # modify1
typeDirPathSave='/home/a504/data/2_test_nm'               # modify2
temp=np.zeros((100,1000))

for pictureDirs in os.listdir(typeDirPath):
    #这是进入了其中一个类别
    pictureDirsRoot=os.path.join(typeDirPath,pictureDirs)
    pictures = []
    for picture in os.listdir(pictureDirsRoot):
        # 这才是进入图片
        pictureRoot=os.path.join(typeDirPath,pictureDirs,picture)
        pictures.append(pictureRoot)
    #排好图片的顺序路径，确保特征是按照时间排序的，否则时间特征将没用
    pictures = sorted(pictures, key=lambda d: int(d.split('(')[-1].split(')')[0]))
    for j in range(100):
        #提取1000维特征
        #这是需要提取特征的图片的路径，每次按顺序提取100张
        image = caffe.io.load_image(pictures[j])
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        output = net.forward()
        ##这段可以打印出你整个caffemode里面每一层的名字和参数个数之类的，可以任意挑取你想要的特征。
        ## obtain the madol name and type 确保你要的那层特征的名字正不正确
        # for layer_name, blob in net.blobs.iteritems():
        # print layer_name + '\t' + str(blob.data.shape)
        ##filters is the feature of the im with (1*1000)size
        ##.data是这一层的输出数据 .param是这一层的参数，详情请参考官方文档
        filters = net.blobs['fc8'].data
        temp[j,:]=filters
    print "%s 这里面这100张图片已经转换完成"%pictureDirsRoot
    np.save(os.path.join(typeDirPathSave,pictureDirs+'.npy'),temp)
print "%s文件夹已经转换完成" % typeDirPath
