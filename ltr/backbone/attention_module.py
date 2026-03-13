import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import constant_init, kaiming_init
from torch.nn.parameter import Parameter

class Multi_Context(nn.Module):
    def __init__(self, inchannels):
        super(Multi_Context, self).__init__()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels * 3, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv2(x)
        return x

class Adaptive_Weight(nn.Module):
    def __init__(self, inchannels):
        super(Adaptive_Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.inchannels = inchannels
        self.fc1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inchannels//4, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        weight = self.relu2(self.fc2(weight))
        weight = self.sigmoid(weight)
        out = x * weight
        return out


class Counter_attention(nn.Module):
    def __init__(self, inchannels):
        super(Counter_attention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.sig = nn.Sigmoid()
        self.mc1 = Multi_Context(inchannels)  # 提取多尺度特征
        self.mc2 = Multi_Context(inchannels)

        self.ada_w1 = Adaptive_Weight(inchannels)  # 自适应权重
        self.ada_w2 = Adaptive_Weight(inchannels)

    def forward(self, assistant, present):

        # assistant是帧， present是事件，mc是提取多尺度特征，ada_w是自适应权重
        mc1 = self.mc1(assistant)
        pr1 = present * self.sig(mc1)
        pr2 = self.conv1(present)
        pr2 = present * self.sig(pr2)
        out1 = pr1 + pr2 + present  # 事件

        mc2 = self.mc2(present)
        as1 = assistant * self.sig(mc2)
        as2 = self.conv2(assistant)
        as2 = assistant * self.sig(as2)
        out2 = as1 + as2 + assistant  # 帧

        # 应用自适应权重
        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)

        out = torch.cat([out1, out2], dim=1)

        return out


class SimplifiedCounterAttention(nn.Module):
    def __init__(self, inchannels, dropout_rate=0.1):
        super().__init__()

        # 基础特征变换（包含BN）
        self.frame_conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)
        )
        self.event_conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)
        )

        # 注意力机制（计算两个模态的自适应权重）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels * 2, inchannels // 4, 1),
            nn.BatchNorm2d(inchannels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(inchannels // 4, 2, 1),  # 输出2个权重值
            nn.Sigmoid()  # 使用Sigmoid而不是Softmax，让两个权重独立
        )

        # 可选的额外正则化
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, frame, event):
        # 特征变换
        frame_feat = self.frame_conv(frame)
        event_feat = self.event_conv(event)

        # 计算自适应权重
        combined = torch.cat([frame_feat, event_feat], dim=1)
        weights = self.attention(combined)  # [B, 2, 1, 1]

        # 分别对两个模态进行加权
        frame_weight = weights[:, 0:1, :, :]  # 帧特征的权重
        event_weight = weights[:, 1:2, :, :]  # 事件特征的权重

        # 加权后的特征
        weighted_frame = frame_feat * frame_weight
        weighted_event = event_feat * event_weight

        # 沿通道维度拼接（与你之前的实验一致）
        fused = torch.cat([weighted_frame, weighted_event], dim=1)

        # 最终的可选dropout
        fused = self.dropout(fused)

        return fused

class Temporal_Attention(nn.Module):
    """Temporal-wise Attention Layer"""

    def __init__(self, timeWindows=4, reduction=2):
        super(Temporal_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.temporal_excitation = nn.Sequential(nn.Linear(timeWindows, int(timeWindows // reduction)),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(int(timeWindows // reduction), timeWindows),
                                                 nn.Sigmoid()
                                                 )

    def forward(self, input):
        # input: [T, B, C, H, W]
        T, B, C, H, W = input.shape

        temp = self.avg_pool(input)  # [T, B, 1, 1, 1]
        temp = temp.view(T, B)  # [T, B]

        temp = self.temporal_excitation(temp.permute(1, 0))  # [B, T]
        # 转置回来并调整形状
        temp = temp.permute(1, 0).view(T, B, 1, 1, 1)  # [T, B, 1, 1, 1]

        weighted_output = torch.mul(input, temp)  # [T, B, C, H, W]

        # 去掉时间维度
        output = torch.mean(weighted_output, dim=0)

        return output
