# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:28:20 2018

@author: spg
"""

import tensorflow as tf
import numpy as np

from skimage import io,transform
import os
import pandas as pd

w = 224
h= 224
c = 3

test_img_path = "E:/2018/tianchi/guangdong_round1_test_a_20180916"

batch_size = 22

def read_image(path): 
    images = []
    filename = []
    filelist = os.listdir(path)
    
    for file in filelist:
        
        image = io.imread(path +'/'+ file)
        image_resize = transform.resize(image,(w,h,c))
        
        images.append(image_resize)
        
        (filepath,tempfilename) = os.path.split(path +'/'+ file)
        
        filename.append(tempfilename)

    return np.asarray(images,dtype=np.float32), filename 

test_data ,test_data_name = read_image(test_img_path) 
test_img_num = len(test_data)
num_batch = int( test_img_num / batch_size)

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('E:/2018/tianchi/logdir/model.ckpt-900.meta')
    saver.restore(sess,tf.train.latest_checkpoint('E:/2018/tianchi/logdir/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")

    output = []
    temp = []
    for i in range(num_batch):
        start = i * batch_size
        end = start + batch_size
        
        feed_dict = {x:test_data[start:end]}
 
        logits = graph.get_tensor_by_name("y_predict:0")
        classification_result = sess.run(logits,feed_dict)

        predict = tf.argmax(classification_result,1).eval()
        temp = predict.tolist()
        output  = output + temp
        

    label = []
    for i in range(len(output)):
        
        if output[i] == 0:
            label.append('norm')
        else :
            label.append('defect%d'% output[i])

    submission = pd.DataFrame({'filename': test_data_name, 'label': label})
    submission.to_csv('./result.csv', header=None, index=False)