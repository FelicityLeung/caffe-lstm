# -*- coding: utf-8 -*-
#注意：使用这个要在当前py文件下创建checkpoint文件夹，否则会报错
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os
import time
from keras.optimizers import Adam
from keras import optimizers
import data_pre2
import matplotlib.pyplot as plt

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

def train(trainRoot,testRoot,inputshape,nb_classes,saved_model=None,batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    model='lstm'
    data_type='npy'
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))
    # Helper: Stop when we stop learning.Stop when the model loss didn't decrease
    early_stopper = EarlyStopping(patience=5)
    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))
    # Get the data and process it.
    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    _,size=data_pre2.get_files(trainRoot)
    steps_per_epoch = (size* 0.7) // batch_size

    # Get generators.
    generator = data_pre2.read_batchNPY(batch_size, trainRoot)
    val_generator = data_pre2.read_batchNPY(batch_size, testRoot)

    # Get the model.
    rm = lstm(inputshape, nb_classes)

    # Fit!
    # Use fit generator.
    history=rm.model.fit_generator(
    generator=generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nb_epoch,
    verbose=1,
    callbacks=[tb, early_stopper, csv_logger, checkpointer],
    validation_data=val_generator,
    validation_steps=40,
    workers=4)#表示使用线程数
    # #HDF5和其Python库h5py
    # #model.save_weights('my_model_weights.h5')
    # #如果你需要在代码中初始化一个完全相同的模型，请使用：
    # model.load_weights('my_model_weights.h5')
    # #如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine - tune或transfer - learning，你可以通过层名字来加载模型：
    # model.load_weights('my_model_weights.h5', by_name=True)
    # #list all data in history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    x = range(nb_epoch)

    plt.figure(1, figsize=(10, 10))
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.plot(x, train_acc)
    plt.plot(x, val_acc)

    plt.xlabel('Epochs')
    plt.ylabel('loss and accuracy')
    plt.title('train and val')
    plt.grid(True)
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    plt.style.use(['classic'])
    plt.savefig("lstm_keras.png")


def main():
    trainRoot = '/home/a504/PycharmProjects/caffe+lstm/data/train/'
    testRoot = '/home/a504/PycharmProjects/caffe+lstm/data/test/'
    inputshape=(100,1000)
    #这个要与datapre2里面onehot函数的类别相对应
    nb_classes=2
    batch_size=16
    saved_model=None
    nb_epoch = 2
    train(trainRoot, testRoot, inputshape, nb_classes, saved_model, batch_size, nb_epoch)

if __name__ == '__main__':
    main()