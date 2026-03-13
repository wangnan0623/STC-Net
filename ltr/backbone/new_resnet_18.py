import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from .base import Backbone
from ltr.models.backbone import attention_module
from ltr.models.backbone.aihd_early import AIHD_early
from ltr.models.backbone.heat import Heat2D
try:
    from torchvision.models.resnet import model_urls
except ImportError:
    # 对于新版本 torchvision，手动定义 model_urls
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


# BasicBlock适用于restnet 18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Bottleneck适用于resnet 50/101/152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks.
    args:
        block: 指定使用的基本块类型，是BasicBlock还是Bottleneck
        layers: 一个列表，包含每个层中块的数量
        output_layers：指定输出层
        num_classes
        inplanes: 输入通道数
        dilation_factor: 空洞卷积的因子
        frozen_layers: 冻结层的列表"""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 根据给定的层数和膨胀因子动态创建四个残差层
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]

        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        # 定义不同层的特征图步幅
        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        # 根据基础块的类型定义特征输出的通道数
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels


        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # 使用自适应平均池化将特征图大小调整为1*1
        # 使用全连接层进行最终分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        # 遍历所有模块并初始化卷积层和批归一化层的权重，以提高训练速度和效果
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None, attention_matrix=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned
         输入经过conv1, bn1, relu, maxpool，然后逐层通过残差层layer1,layer2,layer3,layer4
         在每个层之后，都检查使得需要将输出存入outputs字典，并在符合条件时提前返回"""
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        # x大小为(batch_size, 3, height, width)，其中batch_size = images * sequences
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x大小变为(batch_size, 64, height / 2, width / 2)

        # # 判断 attention_matrix 的形状
        # # 并调整其大小，由(batch_size, 3, height, width)调整为(batch_size, 1, height/2, width/2)
        # if attention_matrix is not None:
        #     # 如果 attention_matrix 的形状为 [images, sequences, channels, height, width]
        #     if len(attention_matrix.size()) == 5:
        #         # reshape the size of attention matrix
        #         # the input attention_matrix is [images, sequences, channels, height, width]
        #         # the input x is [batch_size, channels, height, width]
        #         attention_matrix = attention_matrix.view(-1, attention_matrix.size(2), attention_matrix.size(3), attention_matrix.size(4))
        #
        #     attention_matrix = self.conv2(attention_matrix)
        #     attention_matrix = self.bn2(attention_matrix)
        #     attention_matrix = self.relu(attention_matrix)
        #     attention_matrix = self.maxpool(attention_matrix)
        #     attention_matrix = self.att_layer1(attention_matrix)
        #
        #     # print(x.shape)
        #     # print(attention_matrix.shape)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        # if attention_matrix is not None:
        #     attention_matrix = self.att_layer2(attention_matrix)
        #     x = self.SA1(x, attention_matrix)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        # if attention_matrix is not None:
        #     attention_matrix = self.att_layer3(attention_matrix)
        #     x = self.SA2(x, attention_matrix)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


# 定义一个函数来初始化新模块的权重
def init_new_modules(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def resnet18_new(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    # # 原本
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    # 修改为仅加载可用权重
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(model_dict, strict=False)

    init_new_modules(model)

    return model


