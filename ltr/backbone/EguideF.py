import torch
from torch import nn
# from mmcv.cnn import constant_init, kaiming_init
from mmengine.model import constant_init, kaiming_init


def last_zero_init(m):
    # 初始化模块的最后一层为0
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


# 定义EgF，主要功能是对输入的特征图进行空间和通道上的注意力
class EgF(nn.Module):

    def __init__(self, inplanes, planes):
        super(EgF, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # softmax和softmax_channnel分别用于空间和通道维度上的softmax操作
        self.softmax = nn.Softmax(dim=2)
        self.softmax_channel = nn.Softmax(dim=1)

        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # 参数重置
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_mul_conv)


    def spatial_pool(self, depth_feature):
        # 不仅是空间注意力，还包含一些通道注意力
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature

        # 调整输入特征的形状
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)

        # 生成空间注意力掩码，使用1*1卷积压缩通道维度，调整形状，之后使用softmax
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)

        # 调整空间注意力掩码的形状，以便与输入的特征进行通道加权求和
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)  # 通过矩阵乘法计算上下文特征
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)  # 恢复上下文特征为原始形状

        return context

    def forward(self, x, depth_feature):
        # 在使用中，x是RGB特征，depth_feature是事件特征
        context = self.spatial_pool(depth_feature)  # 调研spatial_pool生成事件的上下文特征
        # 使用channel_mul_conv生成通道乘法项，并应用sigmoid激活函数
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        # 将深度特征图与通道乘法项相乘，生成调整后的特征图
        fea_e = depth_feature * channel_mul_term
        out1 = torch.sigmoid(fea_e)
        # 最终输出。实际上就是用事件数据生成注意力，将其加在rgb上
        out = x * out1 + x

        return out, fea_e
