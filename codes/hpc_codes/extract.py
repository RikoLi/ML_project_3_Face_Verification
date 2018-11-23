from hpc_modules import *
from hpc_fun import *
from PIL import Image

    

# Build feature-extracting net
m_input = model.input
m_output = model.get_layer('dense_1')
pred_model = Model(inputs=m_input, outputs=m_output)

# Open the testing pic
img = Image.open('../dataset/validation_data/500/027.jpg')
img = img.resize((128, 96)) # Old version DeepID input size
img_arr = np.array(img) # Change to a numpy array
img_arr = img_arr[np.newaxis, :, :, :]

# Extract features
vec = pred_model.predict(img_arr, batch_size=1, verbose=1)
print(vec)