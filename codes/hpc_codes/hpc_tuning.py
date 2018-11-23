'''
Fine-tuning for HPC codes.
'''

from hpc_modules import *
from hpc_net_def import * 

# Initialization
def init():
    print('\n---Start fine-tuning!---')
    # Initialize Keras image generator IDG
    train_IDG = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last'
    )
    val_IDG = ImageDataGenerator(data_format='channels_last')

    batch_size = 20                           # Generated batch size
    target_size = (31, 39)                    # Targeted scaling size

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
    return train_generator, validation_generator, batch_size, target_size, train_data_path, val_data_path, pic_classes, train_size, val_size

train_generator, validation_generator, batch_size, target_size, train_data_path, val_data_path, pic_classes, train_size, val_size = init()

# Load model
modelpath = '../models/model.hdf5'
model = load_model(modelpath, compile=False)

# Use new optimizer to compile
opt = Adam(lr=0.000005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit_generator(
    train_generator,
    steps_per_epoch= train_size//batch_size,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= val_size//batch_size+1,
    callbacks=[
        ModelCheckpoint('../models/ft_model.hdf5', verbose=1, save_best_only=True),
        TensorBoard('../logs/deepid/', batch_size=4)
    ]
)