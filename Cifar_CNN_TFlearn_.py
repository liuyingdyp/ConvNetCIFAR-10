from __future__ import  division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical # 混洗 独热
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
# from tflearn.datasets import cifar10

#数据加载和预处理
# Data loading and preprocessing

#读取文件
def load_pickle(f):
    version = platform.python_version_tuple() # 取python版本号
    if version[0] == '2':
        return pickle.load(f) # pick.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('invalid python version: {}'.format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f) # dict 类型
        X = datadict['data']      # X, ndarray, 像素值
        Y = datadict['labels']    # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X , Y

def load_CIFAR10(path):
    """ load all of cifar """
    xs = [] # list
    ys = []

    # 训练集batch 1~5
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X) # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs) # 在list尾部添加对象X, x = [..., [X]]
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

data_path = './cifar-10/cifar-10-batches-py'
X, Y, X_test, Y_test = load_CIFAR10(data_path)
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# 实时数据预处理
# Real-time data preprocessing
img_pre = ImagePreprocessing()
img_pre.add_featurewise_zero_center() # 零中心化,将每个样本的中心设为零，并指定平均值。如果未指定，则对所有样本求平均值。
img_pre.add_featurewise_stdnorm() # STD标准化,按指定的标准偏差缩放每个样本。如果未指定std，则对所有样本数据进行std评估。


# 实时数据扩充
# Real-time data augmentation
img_aug = ImageAugmentation() # 实时数据增强
img_aug.add_random_flip_leftright() # 左右翻转
img_aug.add_random_rotation(max_angle=25.) # 随机旋转

# 卷积网络的建立
# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_pre,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu') # 第一层卷积层。32个卷积核，尺寸3x3，激活函数ReLU
network = max_pool_2d(network, 2) # 最大池化层。滑动窗口步幅为2
network = conv_2d(network, 64, 3, activation='relu') #第二层卷积层，64个卷积核
network = conv_2d(network, 64, 3, activation='relu') #第三层卷积层，64个卷积核
network = max_pool_2d(network, 2) # 最大池化层
network = fully_connected(network, 512, activation='relu') #全连接层，512个神经元
network = dropout(network, 0.5) # 让50%的神经元工作
network = fully_connected(network, 10, activation='softmax') # 全连接层，10个神经元和Softmax激活函数对cifar10分类
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001) # Adam优化器，分类交叉熵损失函数，学习率0.001

# 训练&评估
# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir="./tflearn_logs/")# 实例化，摘要详细级别0（最快）。0:loss accuracy 1:加上gradients 2:加上weights 3:加上activation sparsity
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=96, run_id='cifar10_cnn') # 迭代次数50，打乱数据，显示指标，批量大小96