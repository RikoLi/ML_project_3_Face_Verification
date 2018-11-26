'''
Run on HPC.
'''
from hpc_modules import *

# VGG
def genVGG(pic_classes, shape_tuple):
    '''
    return: Keras Sequential\n
    pic_classes: Pic classes\n
    shape_tuple: Pic size, tuple, (width, height, channel)
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape_tuple))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(pic_classes, activation='softmax'))
    return model

# DeepID + ResNet
def genDeepres(pic_classes, shape_tuple):
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)

    conv1 = Conv2D(64, (7, 7), strides=2, activation='relu')(img_input)
    pool1 = MaxPooling2D((3, 3), strides=2)(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = add([conv2, pool1])
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = add([conv3, conv2])
    conv4 = Activation('relu')(conv3)

    conv4_b = Conv2D(128, (1, 1))(conv3)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = Conv2D(128, (3, 3), padding='same')(conv5)
    conv5 = add([conv5, conv4_b])
    conv5 = Activation('relu')(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv6 = add([conv6, conv5])
    conv8 = Activation('relu')(conv6)

    conv8_b = Conv2D(256, (1, 1))(conv8)

    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)
    conv9 = Conv2D(256, (3, 3), padding='same')(conv9)
    conv9 = add([conv9, conv8_b])
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv10)
    conv10 = add([conv10, conv9])
    conv10 = Activation('relu')(conv10)

    avg_pool = AvgPool2D()(conv10)
    fl1 = Flatten()(avg_pool)
    fl2 = Flatten()(conv10)
    fc = concatenate([fl1, fl2])
    fc = Dense(160, activation='relu')(fc)
    output = Dense(pic_classes, activation='softmax')(fc)

    model = Model(inputs=img_input, outputs=output)
    return model

# DeepID
def genDeepid(pic_classes, shape_tuple):
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    img_input_aug = GaussianNoise(1.0)(img_input)

    conv1 = Conv2D(32, (4,4), activation='relu')(img_input_aug)
    conv1 = Conv2D(32, (4,4), activation='relu', padding='same')(conv1)
    maxpool1 = MaxPooling2D()(conv1)
    maxpool1 = Activation('relu')(maxpool1)
    conv2 = Conv2D(40, (3,3), activation='relu')(maxpool1)
    conv2 = Conv2D(40, (3,3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(40, (3,3), activation='relu', padding='same')(conv2)
    maxpool2 = MaxPooling2D()(conv2)
    maxpool2 = Activation('relu')(maxpool2)
    conv3 = Conv2D(60, (3,3), activation='relu')(maxpool2)
    conv3 = Conv2D(60, (3,3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(60, (3,3), activation='relu', padding='same')(conv3)
    maxpool3 = MaxPooling2D()(conv3)
    maxpool3 = Activation('relu')(maxpool3)
    conv4 = Conv2D(80, (3,3), activation='relu', padding='same')(maxpool3)
    conv4 = Conv2D(80, (3,3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(80, (3,3), activation='relu', padding='same')(conv4)
    maxpool4 = MaxPooling2D()(conv4)
    maxpool4 = Activation('relu')(maxpool4)
    conv5 = Conv2D(100, (3,3), activation='relu', padding='same')(maxpool4)
    conv5 = Conv2D(100, (3,3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(100, (3,3), activation='relu', padding='same')(conv5)
    maxpool5 = MaxPooling2D()(conv5)
    maxpool5 = Activation('relu')(maxpool5)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(maxpool5)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)
    maxpool6 = MaxPooling2D()(conv6)
    maxpool6 = Activation('relu')(maxpool6)
    conv7 = Conv2D(150, (2,2), activation='relu', padding='same')(maxpool6)
    conv7 = Conv2D(150, (2,2), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(150, (2,2), activation='relu', padding='same')(conv7)

    fl1 = Flatten()(conv7)
    fl2 = Flatten()(maxpool6)
    concate1 = concatenate([fl1, fl2])
    concate1 = Dropout(0.5)(concate1)

    fc1 = Dense(160, activation='relu')(concate1)

    output = Dense(pic_classes, activation='softmax')(fc1)

    model = Model(inputs=img_input, outputs=output)
    return model

# ResNet + VGG
def genResvgg(pic_classes, shape_tuple):
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    img_input_aug = BatchNormalization()(img_input)
    # img_input_aug = GaussianNoise(1.0)(img_input_aug)
    branch1 = Conv2D(32, (3,3), padding='same')(img_input_aug)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(32, (3,3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(32, (3,3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    adder1 = Conv2D(32, (1,1), padding='same')(img_input_aug)
    add1 = add([adder1, branch1])
    add1 = Activation('relu')(add1)

    branch2 = Conv2D(64, (3,3), padding='same')(add1)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (3,3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (3,3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    adder2 = Conv2D(64, (1,1), padding='same')(add1)
    add2 = add([adder2, branch2])
    add2 = Activation('relu')(add2)

    branch3 = Conv2D(128, (3,3), padding='same')(add2)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(128, (3,3), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(128, (3,3), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    adder3 = Conv2D(128, (1,1), padding='same')(add2)
    add3 = add([adder3, branch3])
    add3 = Activation('relu')(add3)

    branch4 = Conv2D(256, (3,3), padding='same')(add3)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(256, (3,3), padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(256, (3,3), padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)
    adder4 = Conv2D(256, (1,1), padding='same')(add3)
    add4 = add([adder4, branch4])
    add4 = Activation('relu')(add4)

    pool1 = AvgPool2D()(add4)
    pool1 = Flatten()(pool1)
    fc = Dense(200, activation='relu')(pool1)
    output = Dense(pic_classes, activation='softmax')(fc)

    model = Model(inputs=img_input, outputs=output)
    return model


    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    img_input_aug = BatchNormalization()(img_input)

    branch1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(img_input_aug)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    adder1 = Conv2D(64, (1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(img_input_aug)
    add1 = add([adder1, branch1])
    add1 = Activation('relu')(add1)
    add1 = Dropout(0.4)(add1)

    branch2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(add1)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    adder2 = Conv2D(128, (1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(add1)
    add2 = add([adder2, branch2])
    add2 = Activation('relu')(add2)
    add2 = Dropout(0.4)(add2)

    branch3 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(add2)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    adder3 = Conv2D(256, (1,1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(add2)
    add3 = add([adder3, branch3])
    add3 = Activation('relu')(add3)
    add3 = Dropout(0.4)(add3)

    pool1 = AvgPool2D()(add3)
    pool1 = Flatten()(pool1)
    fc = Dense(512, activation='relu')(pool1)
    output = Dense(pic_classes, activation='softmax')(fc)

    model = Model(inputs=img_input, outputs=output)
    return model

# Inception block
def InceptionBlock(inputLayer):
    inputLayer = BatchNormalization()(inputLayer)
    b1_1 = Conv2D(64, (1,1), padding='same', activation='relu')(inputLayer)
    b1_1 = BatchNormalization()(b1_1)
    b2_1 = Conv2D(96, (1,1), padding='same', activation='relu')(inputLayer)
    b2_1 = BatchNormalization()(b2_1)
    b3_1 = Conv2D(16, (1,1), padding='same', activation='relu')(inputLayer)
    b3_1 = BatchNormalization()(b3_1)
    b4_1 = MaxPooling2D(pool_size=(1,1), strides=1, padding='same')(inputLayer)

    b4_1 = Dropout(0.5)(b4_1)

    b2_2 = Conv2D(128, (3,3), padding='same', activation='relu')(b2_1)
    b2_2 = BatchNormalization()(b2_2)
    b3_2 = Conv2D(32, (3,3), padding='same', activation='relu')(b3_1)
    b3_2 = BatchNormalization()(b3_2)
    b4_2 = Conv2D(32, (1,1), padding='same', activation='relu')(b4_1)
    b4_2 = BatchNormalization()(b4_2)

    b4_2 = Dropout(0.5)(b4_2)

    b3_3 = Conv2D(32, (3,3), padding='same', activation='relu')(b3_2)
    b3_3 = BatchNormalization()(b3_3)

    concate = concatenate([b1_1, b2_2, b3_3, b4_2])
    outputLayer = Activation('relu')(concate)
    return outputLayer

# Inception
def genInception(pic_classes, shape_tuple, isDrawPlot):
    shape_tuple = list(shape_tuple)
    shape_tuple.insert(0, None)
    shape_tuple = tuple(shape_tuple)

    img_input = Input(batch_shape=shape_tuple)
    c1 = Conv2D(32, (3,3), activation='relu')(img_input)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3,3), activation='relu')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D()(c1)

    i1 = InceptionBlock(p1)
    i1 = InceptionBlock(i1)
    i1 = InceptionBlock(i1)

    p2 = MaxPooling2D()(i1)
    p2 = Flatten()(p2)
    fc = Dense(512, activation='relu')(p2)
    output = Dense(pic_classes, activation='softmax')(fc)
    model = Model(inputs=img_input, outputs=output)
    if isDrawPlot:
        plot_model(model, 'inception.png', show_shapes=True, show_layer_names=True)
    model.summary()
    return model