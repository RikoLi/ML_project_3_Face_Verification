from required_modules import *
from net_def import * 

# 初始化Keras图像生成器IDG
train_IDG = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    horizontal_flip=True,
    vertical_flip=False,
    data_format='channels_last'
)
val_IDG = ImageDataGenerator(data_format='channels_last')

batch_size = 64                           # 图像生成器一次产生的batch大小
target_size = (31, 39)                    # 要将图像缩放到的目标size

train_data_path = 'small_train_data'      # 训练集地址，调试用
val_data_path = 'small_val_data'          # 验证集地址，调试用

# 使用生成器生成训练集
print('生成训练集中...')
train_generator = train_IDG.flow_from_directory(
    directory='../../../../dataset/'+train_data_path,
    target_size=target_size,
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)
print('Done!')
print('========================')

# 使用生成器生成验证集
print('生成验证集中...')
validation_generator = val_IDG.flow_from_directory(
    directory='../../../../dataset/'+val_data_path,
    target_size=target_size,
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)
print('Done!')
print('========================')

# 建立网络
model = genDeepID(50, (31, 39, 3))
# model = genResnet(50, (64, 64, 3))
# os.system('pause')

################################################################
# 训练一个小网络测试参数情况
# sgd = SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
adam = Adam(lr=0.0003)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=9108//batch_size,
    epochs=100,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=100//batch_size+1,
    callbacks=[
        ModelCheckpoint('../local_models/test_model_deepid_c50.hdf5', verbose=1, save_best_only=True),
        TensorBoard('../local_logs', batch_size=64, write_images=True)
    ]
)