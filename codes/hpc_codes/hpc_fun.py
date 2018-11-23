'''
Other functions
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Delete some invalid functions on Linux

# Get img index
def readIndex(index_file_name):
    index_list = []
    new_index_list = []
    with open('../pic_index/'+index_file_name, 'r') as f:
        index_list = f.readlines()
    for each in index_list:
        new_str = each.replace('\n', '')
        new_index_list.append(new_str)
    return new_index_list

# Generate data dictionary
def genDataDict(index_list):
    dict_list = []
    for each_str in index_list:
        split_strlist = each_str.split(' ')
        dict_item = {'pic_path':split_strlist[0], 'category':split_strlist[1]}
        dict_list.append(dict_item)
    return dict_list

# Get numpy array of a pic from val data
def path2matr(pic_path):
    img = mpimg.imread('../dataset/validation_data/'+pic_path)
    return img


# 接口使用样例
# index_list = readIndex('train.txt')
# dict_list = genDataDict(index_list)
# pic = dict_list[1000]['pic_path']
# showPic(pic)
