import os
import tensorflow as tf
from practice_own_dataset import decoder
import numpy as np
from keras.utils import to_categorical

img, label = decoder('train.tfrecords')

data = []
label_ = []
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(200):
        example, l = sess.run([img,label])
        data.append(example)
        label_.append(l)
data = np.array(data)
data = np.reshape(data,[200,12288])
label_ = np.array(label_)
label_=to_categorical(label_)
# print(data.shape)
# print(label_)


## comput_accuracy, define weight, bias, conv2d, max_pool ##
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

## define placeholder ##
xs = tf.placeholder(dtype=tf.float32, shape=[None,12288])
ys = tf.placeholder(dtype=tf.float32, shape=[None,2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,64,64,3])

## conv1 layer ##
W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 64x64x32
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 32x32x64
h_pool2 = max_pool_2x2(h_conv2)     # output size 16x16x64


## fc1 layer ##
W_fc1 = weight_variable([16*16*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 16, 16, 64] ->> [n_samples, 16*16*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1])) # loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):

    sess.run(train_step, feed_dict={xs: data, ys: label_, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            data,label_))