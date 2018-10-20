'''
主函数体，用于整个流程脚本的编写
'''
import fun

index_list = fun.readIndex('train.txt')
dict_list = fun.genDataDict(index_list)
pic = dict_list[0]['pic_path']
fun.showPic(pic)