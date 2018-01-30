# -*- coding: utf-8 -*-
"""
这个用于评估所有测试函数的准确率
？有空要补上所有分类的准确率
"""
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
import data_pre2
def lstm(input_shape,nb_classes):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(1024, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    # optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy']
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=metrics)
    return model

def evaluate(inputshape,nb_classes):
    testRoot = '/home/a504/PycharmProjects/caffe+lstm/caffe+lstm/test/'
    model = lstm(inputshape, nb_classes)
    model.load_weights('/home/a504/PycharmProjects/caffe+lstm/data/checkpoints/lstm-npy.009-0.545.hdf5')
    X,Y=data_pre2.read_allTest(testRoot)
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def main():
    inputshape = (100, 1000)
    nb_classes = 2
    evaluate(inputshape,nb_classes)


if __name__ == '__main__':
    main()

