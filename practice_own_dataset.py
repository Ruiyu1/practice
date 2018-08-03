import os
import tensorflow as tf
from PIL import Image
import numpy as np
### image path ###
path = '/Users/ruiyu/PycharmProjects/tensorflow_practicing/img/'


## class ##

def encode(classes):



    writer = tf.python_io.TFRecordWriter('dog_train.tfrecords')


    # class and path ##
    for index,name in enumerate(classes):

        class_path = path + name +'/'

        for img_name in os.listdir(class_path):
            img_path = class_path + img_name

            if img_path == class_path  + '.DS_Store':
                continue

            # a = img_path.split('/')[-1]

            img = Image.open(img_path)
            if img.mode == 'RGBA':


                img = Image.new("RGB", img.size, (255, 255, 255))
            ### check all the pic in jpg form
            # if img.mode == 'RGB':
            #
            #     print(img)
            # else:
            #     print('no')

            img = img.resize((128,128))


            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


classes = ['birds','cats','dogs']


encode(classes)

def decoder(filename):
    data_ = []
    label_ = []
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.cast(image,tf.float32)
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(3000):
            example, l = sess.run([image,label])#在会话中取出image和label
            data_.append(example)
            label_.append(l)

        coord.request_stop()
        coord.join(threads)
    return data_,label_







