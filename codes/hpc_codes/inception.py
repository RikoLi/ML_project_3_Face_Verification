'''
Google Inception structure
'''

from hpc_modules import *
from hpc_net_def import *

# Initialize Keras image generator IDG
train_IDG = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    # height_shift_range=0.1,
    # width_shift_range=0.1,
    # rotation_range=90,
    # shear_range=0.2,
    rescale=1/255
)
val_IDG = ImageDataGenerator(rescale=1/255)

batch_size = 32                           # Generated batch size
target_size = (50, 50)                    # Targeted scaling size

train_data_path = 'train_data'            # Training data path
val_data_path = 'validation_data'         # Validation data path
pic_classes = 10574

train_size = 433041
val_size = 21148

# Generating training data with IMG gernerator
print('Generating the training set...')
train_generator = train_IDG.flow_from_directory(
    directory='../dataset/'+train_data_path,
    target_size=target_size,
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)
print('Done!')
print('========================')

# Generating val data with IMG gernerator
print('Generating the validation set...')
validation_generator = val_IDG.flow_from_directory(
    directory='../dataset/'+val_data_path,
    target_size=target_size,
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)
print('Done!')
print('========================')


# Training
model = genInception(pic_classes, (50,50,3), isDrawPlot=True)
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.save('../models/inception.h5')
model.fit_generator(
    train_generator,
    steps_per_epoch= train_size//batch_size,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= val_size//batch_size+1,
    callbacks=[
        ModelCheckpoint('../models/inception_weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard('../logs/inception/')
    ]
)