'''
ResNet + VGG
Run on HPC.
'''

from hpc_modules import *
from hpc_net_def import * 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize Keras image generator IDG
train_IDG = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False, 
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    rescale=1./225,
    fill_mode='nearest'
)
val_IDG = ImageDataGenerator()

batch_size = 128                           # Generated batch size
target_size = (32, 32)                    # Targeted scaling size

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

# Bulid the net
model = genEasyres(pic_classes, (32, 32, 3))
# model.load_weights('../models/easyres_weights.hdf5', by_name=True)
plot_model(model, 'easyres.png', show_layer_names=True, show_shapes=True)
# model.summary()

# Compile and save model
adam = Adam(lr=0.001)
# model = load_model('../models/easyres.hdf5')
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
save_model(model, filepath='../models/easyres.hdf5')

# Training
model.fit_generator(
    train_generator,
    steps_per_epoch= train_size//batch_size,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= val_size//batch_size+1,
    callbacks=[
        ModelCheckpoint('../models/easyres_weights.hdf5', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard('../logs/easyres/')
    ]
)