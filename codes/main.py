'''
主函数体，用于整个流程脚本的编写
'''
import fun
import numpy as np
from keras.preprocessing.image import *

# 初始化Keras图像生成器IDG
IDG = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    data_format='channels_last'
)

# 定义一些可能用的到的量
pic_height = 112                        # 规定每张图片高度
pic_width = 96                          # 规定每张图片宽度

# 使用生成器生成训练集
x_train_generator = IDG.flow_from_directory(
    directory='../../dataset/train_data',
    target_size=(128, 128),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)

# 使用生成器生成验证集
x_validation_generator = IDG.flow_from_directory(
    directory='../../dataset/validation_data',
    target_size=(128, 128),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)

# To be continued...
