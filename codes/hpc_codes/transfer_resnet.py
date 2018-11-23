from hpc_modules import *
from hpc_net_def import *
from keras.applications.resnet50 import ResNet50

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
target_size = (224, 224)                    # Targeted scaling size

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





# Load pre-trained model ResNet50
resnet = ResNet50(input_shape=(224, 224, 3), include_top=False)
for layer in resnet.layers:
    layer.trainable = False
resnet.load_weights('../download_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Transfer learning & fine tuning
resnet_out = resnet.output
feature_layer = Dense(1024, activatiion='relu')(resnet_out)
output_layer = Dense(10574, activation='softmax')(feature_layer)
transfer_resnet = Model(inputs=resnet.input ,outputs=output_layer)

adam = Adam(lr=0.0001)
transfer_resnet.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Compile & fit
transfer_resnet.fit_generator(
    train_generator,
    steps_per_epoch= train_size//batch_size,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= val_size//batch_size+1,
    callbacks=[
        ModelCheckpoint('../models/transfer_weights.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
        # TensorBoard('../logs/transfer/')
    ]
)