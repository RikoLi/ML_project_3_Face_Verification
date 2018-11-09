'''
主函数体，用于整个流程脚本的编写
'''
import os
from fun import *
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, add, Activation, AvgPool2D
from keras.models import Sequential, Model
from keras.preprocessing.image import *
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

# 初始化Keras图像生成器IDG
train_IDG = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    horizontal_flip=True,
    vertical_flip=False,
    data_format='channels_last'
)
val_IDG = ImageDataGenerator(data_format='channels_last')

# 定义一些可能用的到的量
pic_height = 112                        # 规定每张图片高度
pic_width = 96                          # 规定每张图片宽度
pic_classes = 10574                     # 规定图片种类

# 使用生成器生成训练集
print('生成训练集中...')
train_generator = train_IDG.flow_from_directory(
    directory='../../dataset/train_data',
    target_size=(64, 64),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)
print('Done!')
print('========================')

# 使用生成器生成验证集
print('生成验证集中...')
validation_generator = val_IDG.flow_from_directory(
    directory='../../dataset/validation_data',
    target_size=(64, 64),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)
print('Done!')
print('========================')

# 建立网络
model = genVGG(pic_classes)
model = genResnet(pic_classes)
# os.system('pause')

# 训练
# sgd = SGD(lr=0.001, momentum=0.95, decay=1e-6, nesterov=True)
adam = Adam(lr=0.01)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=6770,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=331,
    callbacks=[ModelCheckpoint('./checkpoint_model.hdf5', verbose=1, save_best_only=True)]
)

model.save('model.hdf5')