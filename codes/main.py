'''
主函数体，用于整个流程脚本的编写
'''
import fun
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import *

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
'''
生成的训练集少了一类，待debug
'''
print('生成训练集中...')
train_generator = train_IDG.flow_from_directory(
    directory='../../dataset/train_data',
    target_size=(128, 128),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)
print('Done!')
print('========================')

# 使用生成器生成验证集
print('生成验证集中...')
validation_generator = val_IDG.flow_from_directory(
    directory='../../dataset/validation_data',
    target_size=(128, 128),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)
print('Done!')
print('========================')

# 建立VGG16 Net  Keras给的样例
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(pic_classes, activation='softmax'))

# 训练
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=25,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=200
)