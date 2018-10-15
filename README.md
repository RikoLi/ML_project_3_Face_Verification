# ML_project_3_Face_Verification
For ZJU Machine learning course group work project 3.
> 注：不定期更新，上次更新为2018.10.16

---

# 说明
本次任务是人脸识别，直接用Deep Learning来做，涉及到以下几个方面：
* 深度学习框架
* 网络结构
* 训练效率（GPU加速、CUDA配置）

## 深度学习框架&环境搭建
我推荐用Keras，Keras是比较新兴的一个深度学习框架，主要特点就是API风格极简，代码可读性强，方便傻瓜式使用。Keras后端使用的是Theano或者TensorFlow，下文给出的环境配置中用的是TensorFlow。我个人也用的是TensorFlow。<del>Google大法好！</del>

Keras开发环境配置一条龙请参考：
* Windows环境搭建：https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/
* Linux环境搭建：https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/

Keras官方文档（使用指南）请参考：
* 中文版：https://keras-cn.readthedocs.io/en/latest/
* 英文版：https://keras.io/

目前中文版已经是最新的英文版翻译了，直接看应该没问题。

## 网络结构选择
FTP上下发的参考文档里给出了几种比较新的网络结构，我们从中具体选择几种（目前估计是两种吧，太多了看不过来）弄明白它们架构的核心思想，然后自行设计一个结合两种思想的网络进行后面的训练。

给出的经典结构参考：
* VGG
* ResNet
* Inception
* DenseNet
* ShuffleNet
* MobileNet
* SENet
* Xception

目前我的想法是先去查一查人脸识别领域的资料，看看有没有推荐的网络结构，然后我们把它和给出的经典结构中的某个结合一下，有机会我们最好讨论一下。

## CUDA配置
CUDA用于给NVIDIA公司的GPU加速提供平台，如果你们的显卡是NVIDIA的而且想尝试GPU加速，请按照如下步骤配置CUDA：
1. 查看显卡支持的CUDA版本：https://jingyan.baidu.com/article/6fb756ec4fabc4241858fbf7.html
1. 确认你的TensorFlow版本支持哪个版本的CUDA，这个自己查吧...把版本匹配好
1. 下载CUDA安装程序：https://developer.nvidia.com/cuda-toolkit-archive
1. 找到你安装CUDA支持的CuDNN，官方下载CuDNN：https://developer.nvidia.com/cudnn 官方安装似乎需要做调查问卷，如果不想就百度网盘啥的看看吧
1. 把CUDA，CuDNN装好即可

折腾CUDA这一条龙挺费时间的，如果你一定要用GPU加速的话，一定要记住：**版本要匹配！！！** 当初我就是版本搞错了...浪费了很多时间在环境配置上...

版本匹配大概是这些东西版本要匹配好：
* CUDA
* CuDNN
* TensorFlow版本，注意TensorFlow分CPU版和GPU版，前面配置Keras的时候有提到
* Keras版本

另外还要注意，如果显卡名是NVIDIA系带M字样的笔记本专用卡，请慎用GPU加速，Keras配置时给出的解释是有烧坏卡的风险...（我的破笔记本就是M系的卡...不过我还是装了CUDA...）

CUDA配置比较麻烦，请务必根据需要与否配置。
