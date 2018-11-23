'''
主函数体，用于整个流程脚本的编写
'''
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

# 定义一些可能用的到的量
# pic_height = 112                        # 规定每张图片高度
# pic_width = 96                          # 规定每张图片宽度
# pic_classes = 10574                     # 规定图片种类
batch_size = 64                           # 图像生成器一次产生的batch大小
target_size = (64, 64)                    # 要将图像缩放到的目标size

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
# model = genVGG(1000, (64, 64, 3))
model = genResnet(1000, (64, 64, 3))

# 训练
# sgd = SGD(lr=0.01, momentum=0.95, decay=1e-6, nesterov=True)
# adam = Adam(lr=0.001)
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=6770,
#     epochs=20,
#     verbose=1,
#     validation_data=validation_generator,
#     validation_steps=331,
#     callbacks=[ModelCheckpoint('../local_models/checkpoint_model.hdf5', verbose=1, save_best_only=True)]
# )





###################################################################
# 训练一个小网络测试参数情况
sgd = SGD(lr=1e-5, momentum=0.95, decay=0, nesterov=True)
# adam = Adam(lr=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=77371//batch_size,
    epochs=20,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=2000//batch_size+1,
    callbacks=[
        ModelCheckpoint('../local_models/small_test_model_resnet.hdf5', verbose=1, save_best_only=True),
        TensorBoard('../local_logs', batch_size=64, write_images=True)
    ]
)
