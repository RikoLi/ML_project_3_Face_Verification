'''
函数、类定义
'''
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, add, Activation, AvgPool2D
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing.image import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 接口使用样例
# index_list = fun.readIndex('train.txt')
# dict_list = fun.genDataDict(index_list)
# pic = dict_list[0]['pic_path']
# fun.showPic(pic)

'''Net generators'''
# 生成VGG16网络
def genVGG(pic_classes):
    '''
    生成一个经典的VGG网络模型\n
    return: Keras Sequential模型
    pic_classes: 输出种类数
    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(pic_classes, activation='softmax'))
    plot_model(model, 'vgg16.png')

    return model

# 生成ResNet网络
def genResnet(pic_classes):
    '''
    生成一个ResNet\n
    return: Keras模型
    '''
    img_input = Input(batch_shape=(None, 64, 64, 3))
    conv1_1 = Conv2D(64, (7, 7), strides=2, activation='relu')(img_input)
    pool1_1 = MaxPooling2D((3, 3), strides=2)(conv1_1)

    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1_1)
    conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_1)
    merge1 = add([pool1_1, conv2_2])
    merge1 = Activation('relu')(merge1)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    conv2_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_3)

    conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_4)
    conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv3_2)
    merge2 = add([conv2_4, conv3_3])
    merge2 = Activation('relu')(merge2)
    merge2 = Conv2D(128, (1, 1), activation='relu', padding='same')(merge2)
    conv3_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
    conv3_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_4)
    merge3 = add([conv3_5, merge2])
    merge3 = Activation('relu')(merge3)

    merge3 = Conv2D(256, (1, 1), activation='relu', padding='same')(merge3)
    conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge3)
    conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_1)
    merge4 = add([conv4_2, merge3])
    merge4 = Activation('relu')(merge4)

    avg_pool = AvgPool2D()(merge4)
    avg_pool = Flatten()(avg_pool)

    fc = Dense(pic_classes, activation='softmax')(avg_pool)
    
    model = Model(inputs=img_input, outputs=fc)
    
    plot_model(model, 'resnet.png')
    return model

'''Functions'''
# 显示数据集中某张图片
def showPic(pic_path):
    '''
    return: none\n
    pic_path: 数据集字典列表中某张图的路径，对应字典键pic_path
    '''
    img = mpimg.imread('E:/study/grade3_winter/Machine_learning/Homework_Undergraduate/Programming_Assignment/Assignment03_FaceVerification/dataset/CASIA-WebFace-Align-96/'+pic_path)
    plt.imshow(img)
    plt.show()

# 将原始数据集转换成一个list，list每个元素为字符串，包括一张图片的相对路径和类别标签
def readIndex(index_file_name):
    '''
    return: list\n
    index_file_name: 数据集索引文件，格式为txt
    '''
    index_list = []
    new_index_list = []
    with open('../temp/'+index_file_name, 'r') as f:
        index_list = f.readlines()
    for each in index_list:
        new_str = each.replace('\n', '')
        new_index_list.append(new_str)
    return new_index_list

# 生成数据字典列表，每个列表元素格式为字典：{pic_path:<pic_path>, category:<pic_category>}
def genDataDict(index_list):
    '''
    return: list\n
    index_list: 由readIndex()生成的list
    '''
    dict_list = []
    for each_str in index_list:
        split_strlist = each_str.split(' ')
        dict_item = {'pic_path':split_strlist[0], 'category':split_strlist[1]}
        dict_list.append(dict_item)
    return dict_list

# 使用matplotlib获取图片矩阵，并以numpy数组形式返回
def path2matr(pic_path):
    '''
    return: numpy数组\n
    pic_path: 数据集字典列表中某张图的路径，对应字典键pic_path
    '''
    img = mpimg.imread('E:/study/grade3_winter/Machine_learning/Homework_Undergraduate/Programming_Assignment/Assignment03_FaceVerification/dataset/CASIA-WebFace-Align-96/'+pic_path)
    return img
