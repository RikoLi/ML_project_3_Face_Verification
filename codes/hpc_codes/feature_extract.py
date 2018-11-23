from hpc_modules import *
from hpc_net_def import *
from hpc_fun import *
from PIL import Image

# Load model
model = load_model('../models/old_models/ft_deepid_model.hdf5')

# Build feature-extracting net
p_input = model.input
p_output = model.get_layer('dense_1').output
extractor = Model(inputs=p_input, outputs=p_output)

# Get test index list
test_list_path = '../dataset/test_lst.csv'
with open(test_list_path, 'r') as f:
    test_list = f.readlines()
    temp = []
    for item in test_list:
        temp.append(item.replace('\n', ''))
test_list = temp
coupleDict_list = []
for item in test_list:
    temp1 = item.split(' ')[0]
    temp2 = item.split(' ')[1]
    coupleDict = {'img1':temp1, 'img2':temp2}
    coupleDict_list.append(coupleDict)

# Resize input images
imgCouple_list = []
counter = 0
for item in coupleDict_list:
    img1_path = '../dataset/test_data/' + item['img1']
    img2_path = '../dataset/test_data/' + item['img2']
    img1 = Image.open(img1_path)
    img1 = img1.resize((128, 96), Image.NEAREST)
    img2 = Image.open(img2_path)
    img2 = img2.resize((128, 96), Image.NEAREST)
    imgCouple = {'img1':img1, 'img2':img2}
    imgCouple_list.append(imgCouple)
    counter += 1
    print('Process: ', counter, '/', len(coupleDict_list))

# Generate image numpy arrays
counter = 0
input_dict_list = []
for item in imgCouple_list:
    # Convert img to matrix...
    mtr1 = np.asarray(item['img1'], 'f')
    mtr1 = mtr1[np.newaxis,:,:,:]
    mtr2 = np.asarray(item['img2'], 'f')
    mtr2 = mtr2[np.newaxis,:,:,:]
    temp = {'img1':mtr1, 'img2':mtr2}
    input_dict_list.append(temp)
    counter += 1
    print('Process: ', counter, '/', len(imgCouple_list))

# Extract features
dist_list = []
counter = 0
for item in input_dict_list:
    counter += 1
    f1 = extractor.predict(item['img1'])
    f2 = extractor.predict(item['img2'])
    dist = np.linalg.norm(f1-f2)
    dist_list.append(dist)
    print('Distance calculating process:', counter, '/', len(input_dict_list))
print('Total distance amounts:', len(dist_list))

# Save distances to a file
print('Saving distances to a file...')
with open('deepid_distance.csv', 'a') as f:
    for item in dist_list:
        f.write(str(item)+'\n')
print('Done!')
