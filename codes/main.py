'''
主函数体，用于整个流程脚本的编写
'''
import fun
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model
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

# 使用生成器生成训练集
'''
生成的训练集少了一类，待debug
'''
print('生成训练集中...')
train_generator = train_IDG.flow_from_directory(
    directory='../../dataset/train_data',
    target_size=(256, 256),
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
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)
print('Done!')
print('========================')

# 建立Inception Net  Keras给的样例
input_img = Input(shape=(256, 256, 3))
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = concatenate([tower_1, tower_2, tower_3], axis=1)

model = Model(inputs=input_img, outputs=output)

# 训练
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=800
)