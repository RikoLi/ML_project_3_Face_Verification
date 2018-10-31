'''
将WebFace大数据集按照train.txt与val.txt分成两个小数据集。
训练集共433041张图片，验证集共21150张图片。
做数据集划分前请修改下面代码中的相对地址，然后运行本文件即可。

结果：
在指定目录下生成训练集和验证集，文件组织结构和原始WebFace的结构一样，
只是把训练用图和验证用图分开了，方便Keras导入。
'''
import fun
import os
import re

# 正则表达式设置，用于文件夹创建
pattern = re.compile(r'/(\d\d\d.jpg)')

# 导入数据集list
train_index_list = fun.readIndex('train.txt')
val_index_list = fun.readIndex('val.txt')
train_directory_list = []
train_label_list = []
train_name_list = []
val_directory_list = []
val_label_list = []
val_name_list = []
for each in train_index_list:
    temp = each.split(' ')
    pic_name = re.findall(pattern, each)[0]
    train_directory_list.append(temp[0].replace('/'+pic_name, ''))
    train_label_list.append(temp[1])
    train_name_list.append(pic_name)
for each in val_index_list:
    temp = each.split(' ')
    pic_name = re.findall(pattern, each)[0]
    val_directory_list.append(temp[0].replace(pic_name, ''))
    val_label_list.append(temp[1])
    val_name_list.append(pic_name)

# 图片划分为训练集和验证集
print('划分训练集中...')
for i in range(len(train_directory_list)):
    if not os.path.exists('../../dataset/train_data/'+train_directory_list[i]):
        os.mkdir('../../dataset/train_data/'+train_directory_list[i])
    with open('../../dataset/CASIA-WebFace-Align-96/'+train_directory_list[i]+'/'+train_name_list[i], 'rb') as fc:
        f = fc.read()   # 复制
        with open('../../dataset/train_data/'+train_directory_list[i]+'/'+train_name_list[i], 'wb') as fp:
            fp.write(f) # 粘贴
    print('Process:', str(i+1)+'/'+str(len(train_directory_list)))
print('训练集划分完成！')
print('========================')
print('划分验证集中...')
for i in range(len(val_directory_list)):
    if not os.path.exists('../../dataset/validation_data/'+val_directory_list[i]):
        os.mkdir('../../dataset/validation_data/'+val_directory_list[i])
    os.mkdir('../../dataset/validation_data/'+val_directory_list[i])
    with open('../../dataset/CASIA-WebFace-Align-96/'+val_directory_list[i]+'/'+val_name_list[i], 'rb') as fc:
        f = fc.read()   # 复制
        with open('../../dataset/val_data/'+val_directory_list[i]+'/'+val_name_list[i], 'wb') as fp:
            fp.write(f) # 粘贴
    print('Process:', str(i+1)+'/'+str(len(val_directory_list)))
print('验证集划分完成！')