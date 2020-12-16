# ssd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

class SSD(nn.Module):
    # SSD网络是由 VGG 网络后街 multibox 卷积层 组成的, 每一个 multibox 层会有如下分支:
    # - 用于class conf scores的卷积层
    # - 用于localization predictions的卷积层
    # - 与priorbox layer相关联, 产生默认的bounding box

    # 参数:
    # phase: test/train
    # size: 输入图片的尺寸
    # base: VGG16的层
    # extras: 将输出结果送到multibox loc和conf layers的额外的层
    # head: "multibox head", 包含一系列的loc和conf卷积层.

    def __init__(self, phase, size, base, extras, head, num_classes):
        # super(SSD, self) 首先找到 SSD 的父类, 然后把类SSD的对象转换为父类的对象
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg) # layers/functions/prior_box.py class PriorBox(object)
        self.priors = Variable(self.priorbox.forward(), volatile=True) # from torch.autograd import Variable
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512,20)  # layers/modules/l2norm.py class L2Norm(nn.Module)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0]) # head = (loc_layers, conf_layers)
        self.conf = nn.ModuleList(head[1])

        if phase = "test":
            self.softmax = nn.Softmax(dim=-1) # 用于囧穿概率
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) #  layers/functions/detection.py class Detect
            # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.
    def forward(self, x):
        # 定义forward函数, 将设计好的layers和ops应用到输入图片 x 上

        # 参数: x, 输入的batch 图片, Shape: [batch, 3, 300, 300]

        # 返回值: 取决于不同阶段
        # test: 预测的类别标签, confidence score, 以及相关的location.
        #       Shape: [batch, topk, 7]
        # train: 关于以下输出的元素组成的列表
        #       1: confidence layers, Shape: [batch*num_priors, num_classes]
        #       2: localization layers, Shape: [batch, num_priors*4]
        #       3: priorbox layers, Shape: [2, num_priors*4]
        sources = list() # 这个列表存储的是参与预测的卷积层的输出, 也就是原文中那6个指定的卷积层
        loc = list() # 用于存储预测的边框信息
        conf = list() # 用于存储预测的类别信息

        # 计算vgg直到conv4_3的relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s) # 将 conv4_3 的特征层输出添加到 sources 中, 后面会根据 sources 中的元素进行预测

        # 将vgg应用到fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x) # 同理, 添加到 sources 列表中

        # 计算extras layers, 并且将结果存储到sources列表中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True) # import torch.nn.functional as F
            if k % 2 = 1: # 在extras_layers中, 第1,3,5,7,9(从第0开始)的卷积层的输出会用于预测box位置和类别, 因此, 将其添加到 sources列表中
                sources.append(x)

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # apply multibox to source layers

        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
            # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
        # cat 是 concatenate 的缩写, view返回一个新的tensor, 具有相同的数据但是不同的size, 类似于numpy的reshape
        # 在调用view之前, 需要先调用contiguous
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]

        if self.phase == "test":
            # 这里用到了 detect 对象, 该对象主要由于接预测出来的结果进行解析, 以获得方便可视化的边框坐标和类别编号, 具体实现会在后文讨论.
            output = self.detect(
                loc.view(loc.size(0), -1, 4), #  又将shape转换成: [batch, num_boxes, 4], 即[1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)), # 同理,  shape 为[batch, num_boxes, num_classes], 即 [1, 8732, 21]
                self.priors.type(type(x.data))
                # 利用 PriorBox对象获取特征图谱上的 default box, 该参数的shape为: [8732,4]. 关于生成 default box 的方法实际上很简单, 类似于 anchor box, 详细的代码实现会在后文解析.
                # 这里的 self.priors.type(type(x.data)) 与 self.priors 就结果而言完全等价(自己试验过了), 但是为什么?
            )
        if self.phase == "train": # 如果是训练阶段, 则无需解析预测结果, 直接返回然后求损失.
            output = (
                loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
            )
        return output
    def load_weights(self, base_file): # 加载权重文件
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported")

def vgg(cfg, i, batch_norm = False):
    # cfg = base['300'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    # i = 3
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        if v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Con2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

def add_extras():
    exts1_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    exts1_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    exts2_1 = nn.Conv2d(512, 128, 1, 1, 0)
    exts2_2 = nn.Conv2d(128, 256, 3, 2, 1)
    exts3_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts3_2 = nn.Conv2d(128, 256, 3, 1, 0)
    exts4_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts4_2 = nn.Conv2d(128, 256, 3, 1, 0)

    return [exts1_1, exts1_2, exts2_1, exts2_2, exts3_1, exts3_2, exts4_1, exts4_2]


def multibox(vgg, extras, num_classes):
    loc_layers = []
    conf_layers = []
    #vgg_source=[21, -2] # 21 denote conv4_3, -2 denote conv7

    # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
    loc1 = nn.Conv2d(vgg[21].out_channels, 4*4, 3, 1, 1) # 利用conv4_3的特征图谱, 也就是 vgg 网络 List 中的第 21 个元素的输出(注意不是第21层, 因为这中间还包含了不带参数的池化层).
    loc2 = nn.Conv2d(vgg[-2].out_channels, 6*4, 3, 1, 1) # Conv7
    loc3 = nn.Conv2d(vgg[1].out_channels, 6*4, 3, 1, 1) # exts1_2
    loc4 = nn.Conv2d(extras[3].out_channels, 6*4, 3, 1, 1) # exts2_2
    loc5 = nn.Conv2d(extras[5].out_channels, 4*4, 3, 1, 1) # exts3_2
    loc6 = nn.Conv2d(extras[7].out_channels, 4*4, 3, 1, 1) # exts4_2
    loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]

    # 定义分类层, 和定位层差不多, 只不过输出的通道数不一样, 因为对于每一个像素点上的每一个default box,
    # 都需要预测出属于任意一个类的概率, 因此通道数为 default box 的数量乘以类别数.
    conf1 = nn.Conv2d(vgg[21].out_channels, 4*num_classes, 3, 1, 1)
    conf2 = nn.Conv2d(vgg[-2].out_channels, 6*num_classes, 3, 1, 1)
    conf3 = nn.Conv2d(extras[1].out_channels, 6*num_classes, 3, 1, 1)
    conf4 = nn.Conv2d(extras[3].out_channels, 6*num_classes, 3, 1, 1)
    conf5 = nn.Conv2d(extras[5].out_channels, 4*num_classes, 3, 1, 1)
    conf6 = nn.Conv2d(extras[7].out_channels, 4*num_classes, 3, 1, 1)
    conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]

    # loc_layers: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
    # conf_layers: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
    # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W], 上面的顺序有点错误, 不过改起来太麻烦
    return loc_layers, conf_layers

base = {  #vgg网络结构参数
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '500': []
}
extras = { #extras层参数
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '500': []
}
mbox = {  #multibox相关参数
    '300': [4, 6, 6, 6, 4, 4],
    '500': []
}

def build_ssd(phase,size = 300,num_classes = 21):
    # 构建模型参数
    if phase !="test" and phase !="train":   #训练或者预测阶段
        print("ERROR: Phase: "+phase+" not recognized")
        return
    if size !=300:
        print("ERROR: You specified size "+ repr(size)+".Only SSD300 is supported")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size), 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase,size,base_,extras_,head_,num_classes)




