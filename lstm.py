# -*- coding:utf-8 -*-
"""
my tensorflow lstm example
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import data_pre2
import os
#因为tensorflow的版本兼容有各种问题，应该怎么办？
# 设置 GPU 按需增长
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
learning_rate = 0.001
training_steps = 10000
batch_size = 16
display_step = 200
# Network Parameters
num_input = 1000 #data input (img shape: [batchsize,timestep,feature])
timesteps = 100 # timesteps
num_hidden = 128 # hidden layer num
num_classes = 2 #total classes (0-9 digits)


# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
Outputs=[]
def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #这个是每个时间步输出一个状态，如果想要最后一个时间outputs[-1]
    # Outputs.append(outputs)
    # Linear activation, using rnn inner loop last output
    # 这个就是应该直接返回最后一个class10的输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
# print(logits)#Tensor("add:0", shape=(?, 10), dtype=float32)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
#save
saver=tf.train.Saver(tf.global_variables())
# Start training
root='/home/a504/PycharmProjects/caffe+lstm/caffe+lstm/train/'
root2='/home/a504/PycharmProjects/caffe+lstm/caffe+lstm/test/'
# with tf.Session() as sess:
#     # Run the initializer
#     sess.run(init)
#     for step in range(1, training_steps+1):
#         batch_x, batch_y = data_pre2.read_batch(batch_size,root)
#         # Reshape data
#         # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
#         # Run optimization op (backprop)
#         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#         if step % display_step == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
#                                                                  Y: batch_y})
#             print("save ,model：", saver.save(sess, './test_Lstm3.model'))
#             print("Step " + str(step) + ", Minibatch Loss= " + \
#                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.3f}".format(acc))
#test the data by batch acc
#             test_len = 32
#             test_data, test_label = data_pre2.read_batch(test_len, root2)
#             print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
#     print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images


# test the model one by one
with tf.Session() as sess:
    # 参数恢复
    testAcc=0.0
    module_file = tf.train.latest_checkpoint('test_Lstm3.model')
    saver.restore(sess, 'test_Lstm3.model')
    test_len = 1
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    testNum=0.0
    for i in os.listdir(root2):
        for j in os.listdir(os.path.join(root2,i)):
            test_data=data_pre2.read_image(os.path.join(root2,i,j))
            test_data=np.reshape(test_data,(1,100,1000))
            test_label=data_pre2.onehot(int(i[0]))
            test_label = np.reshape(test_label,(1,2))
            acc=sess.run(correct_pred, feed_dict={X: test_data, Y: test_label})
            if acc==True:testAcc+=1.0
            testNum+=1
    print ("test capacity:%d,test right num:%d",testNum,testAcc)
    print("Testing Accuracy:", testAcc/testNum)
