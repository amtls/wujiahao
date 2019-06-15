# 基于深度学习的银行卡号识别

**目录 (Table of Contents)**

[TOCM]

[TOC]

## 项目实现技术栈介绍
### 开发软件
> Visual Studio 2017

### 开发语言
> python3.6

### 使用框架
> Keras2.2.4、opencv-python3.3.0.10、tensorflow1.13.1

### 使用算法
> CTPN、CRNN

## 项目实现思路介绍
### 整体实现思路
#### 银行卡号定位实现
> 1. 首先，用 VGG16 的前 5 个 Conv stage 得到 feature map，大小为  W*H*C
> 2. 用 3*3 的滑动窗口在前一步得到的 feature map 上提取特征，利用这些特征来对多个 anchor 进行预测,这里 > anchor 定义与之前 faster-rcnn 中的定义相同，也就是帮我们去界定出目标待选区域。
> 3. 将上一步得到的特征输入到一个双向的 LSTM 中，输出 W*256 的结果，再将这个结果输入到一个 512 维的> 全连接层（FC）.
>4. 最后通过分类或回归得到的输出主要分为三部分，根据上图从上到下依次为 2k vertical coordinates:表示选择框的高度和中心的 y 轴的坐标；2k scores:表示 的是 k 个 anchor 的类别信息，说明其是否为字符； k side-refinement 表示的是 选择框的水平偏移量。本文实验中 anchor 的水平宽度都是 16 个像素不变， 也就是说我们微分的最小选择框的单位是 “16 像素”。
>5. 用文本构造的算法，将我们得到的细长的矩形，然后将其合并 成文本的序列框

#### 不定长银行卡号识别实现
> 1.传统的ocr识别过程分为两步：单字切割和分类任务，我们一般都会将一连串文字的文本文件先利用投影法切割出单个字体，在送入CNN里进行文字分类。但是通过ctpn算法定位到的卡号图片文本长度不同，所以我们不需要显式加入文字切割这个环节，而是将文字识别转化为序列学习问题，虽然输入的图像尺度不同，文本长度不同，但是经过DCNN和RNN后，在输出阶段经过一定的翻译后，就可以对整个文本图像进行识别，也就是说，文字的切割也被融入到深度学习中去了。
>2.CRNN算法是把CNN做图像特征工程的潜力与LSTM做序列化识别的潜力，进行结合。它既提取了鲁棒特征，又通过序列识别避免了传统算法中难度极高的单字符切分与单字符识别，同时序列化识别也嵌入时序依赖（隐含利用语料）。在训练阶段，CRNN将训练图像统一缩放100×32（w × h）；在测试阶段，针对字符拉伸导致识别率降低的问题，CRNN保持输入图像尺寸比例，但是图像高度还是必须统一为32个像素，卷积特征图的尺寸动态决定LSTM时序长度。

## 项目实现流程介绍
### 卡号定位
#### Detecting Text in Fine-scale proposals（选择出 anchor， 也就是待选的”矩形微分框“）
> 和  faster-rcnn 中的 RPN 的主要区别在于引入了”微分“思想，将我们的的候选区域 切成长条形的框来进行处理。k 个 anchor（也就是 k 个待选的长条预选区域）的设置如 下：宽度都是 16 像素，高度从 11~273 像素变化（每次乘以 1.4），也就是说 k 的值设定为 10。最后结果如下： 

![ "微分"示意图](https://github.com/amtls/MyImages/blob/master/%E5%9B%BE%E7%89%871.jpg?raw=true)
#### Recurrent Connectionist Text Proposals（双向 LSTM，利 用上下文本信息的 RNN 过程）
> 本文使用的方法回归出来的 y 轴坐标结果如下： 

$$v_c = (c_y - c^a _y)/h^a$$

$$v_h = log(h/h^a)$$

$$c^*_c = (c^*_y - c^a_y)/h^a$$

$$v^*_h = log(h^*/h^a)$$
>其中标记*的表示为真值； v = {vc,vh } 表示一个预测的框选位置，因为长度固定（之前确定的16像素），vc表示的是该预选框在y轴上的中心位置，vh表示这个预选框的高度。 
> 其方法对应的就是之前流程中的”双向LSTM“对应的细节，将前后文的信息用到 文本位置的定位当中。其中BLSTM有128个隐含层。输入为3*3*C滑动窗口的feature，输 出为每个窗口所对应的256维的特征。简要表示如下:

![Diagram](https://github.com/amtls/MyImages/blob/master/%E8%BE%93%E5%85%A5%E5%92%8C%E8%BE%93%E5%87%BA.png?raw=true)

#### Side-refinement（文本构造，将多个 proposal 合并成直 线）
>先进行文本位置的构造，Side-refinement是最后进行优化的方法。对定位出来的 “小矩形框”加以合并和归纳，可以得到需要的文本信息的位置信息。我们最后保留的 小矩形框是需要score>0.7的情况，也就是将下图中的红色小矩形框合并，最后生成黄色 的大矩形框
>
![小区域分割示意图](https://github.com/amtls/MyImages/blob/master/%E5%9B%BE%E7%89%872.jpg?raw=true)
>主要的思路为：每两个相近的proposal（也就是候选区）组成一个 pair，合并不同
的 pair直到无法再合并为止。而判断两个proposal,Bi和Bj可以组成一个pair的条件为Bi—  >Bj,同时Bj—>Bi;该符号的判定条件见下图。

![Diagram](https://github.com/amtls/MyImages/blob/master/%E5%8F%AF%E5%90%88%E5%B9%B6%E5%8C%BA%E5%9F%9F.png?raw=true)
>因为这里规定了回归出来的box的宽度是16个像素，所以会导致一些位置上的误 差，这时候就是Side-refinement发挥作用的时候 了。定义的式子如下： 

$$O = (x_{side} -c^a_x)/w^a$$

$$o^*=(x^*_{side})/w^a$$
>其中带*表示为GroundTruth.。𝑥𝑥𝑠𝑠𝑖𝑖𝑑𝑑 𝑒𝑒 表示回归出来的左边界或者右边界，𝑐𝑐 𝑎𝑎 表示 anchor 中心的横坐标，𝑤𝑤𝑎𝑎是固定的宽度16像素。所以O的定义相当于是一个缩放的比例， 帮助我们去拉伸回归之后的box的结果，从而更好地符合实际文本的位置。纵观整个流程，该方法的最大两点也是在于将RNN引入了文本检测之中，同时将 待检测的结果利用“微分”的思路来减少误差，使用固定宽度的anchor来检测分割成 许多块的proposal.最后合并之后的序列就是我们需要检测的文本区域。
### 卡号识别
#### 识别流程：
>·首先会将图像缩放到 32×W×1 大小
·然后经过CNN后变为 1×（W/4）× 512
·接着针对LSTM，设置 T=(W/4) ， D=512 ，即可将特征输入LSTM。
·LSTM有256个隐藏节点，经过LSTM后变为长度为T × nclass的向量，再经过softmax处理，列向量每个元素代表对应的字符预测概·率，最后再将这个T的预测结果去冗余合并成一个完整识别结果即可。


#### 网络结构：
![RCNN网络结构](https://github.com/amtls/MyImages/blob/master/%E6%9C%AA%E6%A0%87%E9%A2%98-2.jpg?raw=true)
>1.卷积层，使用CNN，作用是从输入图像中提取特征序列;
2.循环层，使用RNN，作用是预测从卷积层获取的特征序列的标签（真实值）分布;
3.转录层，使用CTC，作用是把从循环层获取的标签分布通过去重整合等操作转换成最终的识别结果;

![crnn识别流程](https://github.com/amtls/MyImages/blob/master/%E6%9C%AA%E6%A0%87%E9%A2%98-1.jpg?raw=true)
## 项目中卡号定位与卡号识别结果说明

![大赛资料图片测试](https://github.com/amtls/MyImages/blob/master/%E6%B5%8B%E8%AF%951.jpg?raw=true)

![大赛资料图片测试](https://github.com/amtls/MyImages/blob/master/%E6%B5%8B%E8%AF%952.png?raw=true)

![额外图片测试](https://github.com/amtls/MyImages/blob/master/%E6%B5%8B%E8%AF%953.png?raw=true)

## 项目代码介绍文档
##### 项目部署
>bank_card_ocr
	>>bin
	>>crnn:卡号识别
	>>>models：模型文件
	>>>>_init_.py
	>>>>crnn.py
	>>>>
	>>>to_lmdb
	>>>>base.py
	>>>>base64.py
	>>>>dataset.py
	>>>>handle_images.py
	>>>>ocr.py:卡号定位
	>>>>test.py：卡号定位测试
	>>>>train.py：模型训练
	>>>>utils.py：工具集
	>>data：额外测试数据
	models
	test_images：测试图片集
	test_result：测试结果集
	tool
	demo.py：项目启动文件
	config.py：项目配置文件
	difflib.py：
##### 项目运行
>运行项目目录下的demo.py文件
>或者运行打包的demo.exe文件


