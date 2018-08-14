import os
import tensorflow as tf
from PIL import Image
import numpy as np

def encode(classes):
    path = '/Users/ruiyu/PycharmProjects/tensorflow_practicing/pic/'
    writer = tf.python_io.TFRecordWriter('train.tfrecords')

    for index,name in enumerate(classes):

        class_path = path + name +'/'

        for img_name in os.listdir(class_path):
            img_path = class_path + img_name

            if img_path == class_path  + '.DS_Store':
                continue

            img = Image.open(img_path)

            # if img.size[0]<64 or img.size[1]<64:
            #     print(img.size)

            # print(index)
            if img.mode == 'RGBA':

                img = img.convert('RGB')

            img = img.resize((64,64))

            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()
# #
classes = [ 'cats', 'dogs']
# encode(classes)

def decoder(filename):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    img = tf.decode_raw(features['img_raw'],tf.uint8)

    img = tf.reshape(img, [64,64,3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img,label

# img, label = decoder('train.tfrecords')
#
#

# img, label = decoder('train.tfrecords')
# with tf.Session() as sess: #开始一个会话
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     for i in range(200):
#         example, l = sess.run([img,label])
#         # example = np.reshape(example,[64,64,3])
#
#         print(l)


