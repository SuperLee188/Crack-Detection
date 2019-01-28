# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:54:40 2018

@author: SuperLee
"""

# -*- coding :utf-8 -*-

import tensorflow as tf
from skimage import io,transform
import glob
import numpy as np
import os

#定义图片的大小为32*32*1  1代表图像是黑白的，只有一个色彩通道
w=224
h=224
c= 3
batch_size = 36
regularaztion_rate = 0.001
training_steps = 901    #大概10个epoch


train_path = "E:/2018/tianchi/guangdong_round1_train2_20180916/"
test_path = "E:/2018/tianchi/guangdong_round1_test_a_20180916/"

LOG_DIR = "logdir"

def read_image(path):
    
    filename = os.listdir(path)
    file = [int(x) for x in filename] #将文件夹名称改为整数后排序
    file_list = sorted(file)
    filedir = ['%d'% x for x in file_list]   #排序后转为字符数 
    label_dir=[path+x for x in filedir if os.path.isdir(path+x)]
    images=[]
    labels=[]
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.jpg'):  #img是图片的路径加上后缀名
            print("reading the img:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(w,h,c))
#           image = tf.image.per_image_standardization(image)
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32) 

#获取训练数据
train_data,train_label =read_image(train_path)


#打乱数据顺序以防训练的模型产生过拟合问题
train_image_num = len(train_data)

one_epoch = int(train_image_num/batch_size)+1  # 60

train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label =train_label[train_image_index]

#定义前向传播的过程以及神经网络的参数
#第一层卷积层，卷积核大小为5*5，深度为32
#输入数据图像的大小为 224*224*3

#定义一个用来显示网络每层结构的数据的函数，显示每一个卷积层或池化层输出的tensor的尺寸
def print_activation(t):
    print(t.op.name, '',t.get_shape().as_list())


def inference (input_tensor,train,regularizer):
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable('weight',[11,11,c,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,4,4,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        variable_summaries(conv1_weights,'weights')
        variable_summaries(conv1_biases,'biases')
        print_activation(relu1)
        
#第二层：池化层，过滤器的尺寸为3*3，使用全0补充，步长为2
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
        print_activation(pool1)


    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable('weight',[5,5,64,192],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[192],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
        variable_summaries(conv2_weights,'weights')
        variable_summaries(conv2_biases, 'biases')
        
        print_activation(relu2)
        

    with tf.variable_scope('pool2'):
        
        
        pool2 = tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
        print_activation(pool2)
        
        
    with tf.variable_scope('conv3'):
        conv3_weights = tf.get_variable('weight',[3,3,192,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias',[256],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
        variable_summaries(conv3_weights,'weights')
        variable_summaries(conv3_biases, 'biases')
        
        print_activation(relu3)
        
        
    with tf.variable_scope('conv4'):    
        conv4_weights = tf.get_variable('weight',[3,3,256,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias',[256],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
        variable_summaries(conv4_weights,'weights')
        variable_summaries(conv4_biases, 'biases')
        
        print_activation(relu4)
        
    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(relu4,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
        print_activation(pool3)
        
        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped = tf.reshape(pool3,[-1,nodes])

#第五层：全连接层，nodes=5×5×16=400，400->120的全连接
#尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
#训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
#这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,4096],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[4096],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)
        variable_summaries(fc1_weights,'weights')
        variable_summaries(fc1_biases, 'biases')

#第六层：全连接层，120->84的全连接
#尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable('weight',[4096,1000],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[1000],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2,0.5)
        variable_summaries(fc2_weights,'weights')
        variable_summaries(fc2_biases,'biases')    
        
            
#第七层：全连接层（近似表示），84->2的全连接
#尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×2。最后，64×2的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
#即得到最后的分类结果。
    with tf.variable_scope('fc3'):
        fc3_weights = tf.get_variable('weight',[1000,12],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[12],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases
        variable_summaries(fc3_weights,'weights')
        variable_summaries(fc3_biases,'biases')
    return logit

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+ name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/' + name ,stddev)
#神经网络至此结束，下面是训练参数配置
def train():
    with tf.name_scope(''):
        x = tf.placeholder(tf.float32,[None,w,h,c],name ='x')   #None表示一个batch内的样例个数
        y_ =tf.placeholder(tf.int32,[None],name ='y_')
        regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
        y = inference (x,False,regularizer)
        
        
       # image_shaped_input = tf.reshape(x,[-1,32,32,1])
       # tf.summary.image('input',image_shaped_input,10) #在tensorboard中随机显示10个
        
        b = tf.constant(value=1,dtype=tf.float32)
        y_predict = tf.multiply(y,b ,name = 'y_predict')

#定义损失函数，学习率，滑动平均操作以及训练过程
#    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    
#    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) #总损失等于交叉熵损失加上正则化损失和
        tf.summary.scalar('loss',loss)
    
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()    
    

    with tf.Session() as sess:
        
        saver = tf.train.Saver()
        
        writer =tf.summary.FileWriter(LOG_DIR,sess.graph)#初始化写日志的writer,并将当前的计算图写入日志        
        
        tf.global_variables_initializer().run()
        acc = 0
        for i in range(0,training_steps):
            start = (i*batch_size) % train_image_num
            end = min(start+batch_size,train_image_num)
            _,train_accuracy,loss_value, summary = sess.run([train_op ,accuracy,loss ,merged],feed_dict ={x:train_data[start:end],y_:train_label[start:end]})
            
            acc += train_accuracy
            
            #test_accuracy = sess.run(accuracy,feed_dict={x:test_data[:80],y_:test_label[:80]})
            
            if i % one_epoch == 0:
                print("After %d epochs ,loss on training data is %g." % (i/one_epoch, loss_value))
                print("After %d epochs ,train accuracy is %g." %( i/one_epoch ,train_accuracy))
                    
                writer.add_summary(summary,i)
            if i % 450 == 0:      
                saver.save(sess,os.path.join(LOG_DIR,"model.ckpt"), i)
        average_acc  = acc/training_steps
        print("After %d epochs ,the average_acc is %g " % (i/one_epoch,average_acc))
     
    writer.close()

def main(argv=None):
    train()

if __name__ =='__main__':
    tf.app.run()
