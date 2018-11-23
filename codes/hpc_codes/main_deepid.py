'''
DeepID
Run on HPC.
'''

from hpc_modules import *
from hpc_net_def import * 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize Keras image generator IDG
train_IDG = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)
val_IDG = ImageDataGenerator()

batch_size = 100                           # Generated batch size
target_size = (90, 110)                    # Targeted scaling size

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
model = genDeepid(pic_classes, (90, 110, 3))
plot_model(model, 'deepid.png', show_layer_names=True, show_shapes=True)

# Compile and save model
# sgd = SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
save_model(model, filepath='../models/deepid.hdf5')

# Training
# model = load_model('../models/deepid.hdf5')
model.fit_generator(
    train_generator,
    steps_per_epoch= train_size//batch_size,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= val_size//batch_size+1,
    callbacks=[
        ModelCheckpoint('../models/deepid_weights.hdf5', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard('../logs/deepid/')
    ]
)