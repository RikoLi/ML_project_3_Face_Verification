'''
生成特定结构的网络函数定义
'''
from required_modules import *

# 生成VGG网络
def genVGG(pic_classes, shape_tuple):
    '''
    生成一个经典的VGG网络模型\n
    return: Keras Sequential模型\n
    pic_classes: 输出种类数\n
    shape_tuple: 图片尺寸tuple，(width, height, channel)
    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=shape_tuple))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(pic_classes, activation='softmax'))
    plot_model(model, '../local_model_plots/vgg.png', show_layer_names=True, show_shapes=True)

    return model

# 生成ResNet网络
def genResnet(pic_classes, shape_tuple):
    '''
    生成一个ResNet\n
    return: Keras模型\n
    pic_classes: 输出种类数
    '''
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    conv1 = Conv2D(16, (7, 7), strides=2, activation='relu')(img_input)
    pool1 = MaxPooling2D((3, 3), strides=2)(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    merge1 = add([pool1, conv3])
    merge1 = Activation('relu')(merge1)

    conv4_branch = Conv2D(64, (1, 1), activation='relu', padding='same')(merge1)
    conv4_branch = BatchNormalization()(conv4_branch)
    conv4_branch = Dropout(0.5)(conv4_branch)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    merge2 = add([conv5, conv4_branch])
    merge2 = Activation('relu')(merge2)

    conv6_branch = Conv2D(128, (1, 1), activation='relu', padding='same')(merge2)
    conv6_branch = BatchNormalization()(conv6_branch)
    conv6_branch = Dropout(0.5)(conv6_branch)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    merge3 = add([conv7, conv6_branch])
    merge3 = Activation('relu')(merge3)

    avg_pool = AvgPool2D()(merge3)
    # avg_pool = AvgPool2D()(merge2)
    avg_pool = Flatten()(avg_pool)

    fc = Dense(pic_classes, activation='softmax')(avg_pool)
    
    model = Model(inputs=img_input, outputs=fc)
    
    plot_model(model, '../local_model_plots/resnet.png', show_layer_names=True, show_shapes=True)
    return model

# 生成DeepID网络
def genDeepID(pic_classes, shape_tuple):
    '''
    生成一个DeepID网络模型\n
    return: Keras Model模型\n
    pic_classes: 输出种类数\n
    shape_tuple: 图片尺寸tuple，(width, height, channel)
    '''
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    img_input_aug = GaussianNoise(1.0)(img_input)

    conv1 = Conv2D(20, (4,4), activation='relu', kernel_regularizer=l2())(img_input_aug)
    maxpool1 = MaxPooling2D()(conv1)
    maxpool1 = Activation('relu')(maxpool1)
    conv2 = Conv2D(40, (3,3), activation='relu', kernel_regularizer=l2())(maxpool1)
    maxpool2 = MaxPooling2D()(conv2)
    maxpool2 = Activation('relu')(maxpool2)
    conv3 = Conv2D(60, (3,3), activation='relu', kernel_regularizer=l2())(maxpool2)
    maxpool3 = MaxPooling2D()(conv3)
    maxpool3 = Activation('relu')(maxpool3)
    conv4 = Conv2D(80, (2,2), activation='relu', kernel_regularizer=l2())(maxpool3)

    fl1 = Flatten()(conv4)
    fl2 = Flatten()(maxpool3)
    concate1 = concatenate([fl1, fl2])
    concate1 = Dropout(0.5)(concate1)

    fc1 = Dense(160, activation='relu')(concate1)

    output = Dense(pic_classes, activation='softmax')(fc1)

    model = Model(inputs=img_input, outputs=output)
    plot_model(model, to_file='../local_model_plots/DeepID.png', show_layer_names=True, show_shapes=True)
    return model