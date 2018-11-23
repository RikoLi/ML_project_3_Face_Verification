'''
杂项函数定义
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 显示数据集中某张图片
def showPic(pic_path):
    '''
    return: none\n
    pic_path: 数据集字典列表中某张图的路径，对应字典键pic_path
    '''
    img = mpimg.imread('E:/study/grade3_winter/Machine_learning/Homework_Undergraduate/Programming_Assignment/Assignment03_FaceVerification/dataset/CASIA-WebFace-Align-96/'+pic_path)
    plt.imshow(img)
    plt.show()

# 将原始数据集转换成一个list，list每个元素为字符串，包括一张图片的相对路径和类别标签
def readIndex(index_file_name):
    '''
    return: list\n
    index_file_name: 数据集索引文件，格式为txt
    '''
    index_list = []
    new_index_list = []
    with open('../../../temp/'+index_file_name, 'r') as f:
        index_list = f.readlines()
    for each in index_list:
        new_str = each.replace('\n', '')
        new_index_list.append(new_str)
    return new_index_list

# 生成数据字典列表，每个列表元素格式为字典：{pic_path:<pic_path>, category:<pic_category>}
def genDataDict(index_list):
    '''
    return: list\n
    index_list: 由readIndex()生成的list
    '''
    dict_list = []
    for each_str in index_list:
        split_strlist = each_str.split(' ')
        dict_item = {'pic_path':split_strlist[0], 'category':split_strlist[1]}
        dict_list.append(dict_item)
    return dict_list

# 使用matplotlib获取图片矩阵，并以numpy数组形式返回
def path2matr(pic_path):
    '''
    return: numpy数组\n
    pic_path: 数据集字典列表中某张图的路径，对应字典键pic_path
    '''
    img = mpimg.imread('E:/study/grade3_winter/Machine_learning/Homework_Undergraduate/Programming_Assignment/Assignment03_FaceVerification/dataset/CASIA-WebFace-Align-96/'+pic_path)
    return img


# 接口使用样例
# index_list = readIndex('train.txt')
# dict_list = genDataDict(index_list)
# pic = dict_list[1000]['pic_path']
# showPic(pic)
