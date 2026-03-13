import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Frequency_Fuse(nn.Module):
    def __init__(self, channelin):
        super().__init__()
        self.norm1 = nn.LayerNorm(channelin)

    def forward(self, frame, event):
        # frame, event : [B, C, H, W]
        # 对输入张量进行二维快速傅里叶变换，得到频域表示
        B, C, H, W = frame.shape

        frame_fft = torch.fft.rfft2(frame.float())
        event_fft = torch.fft.rfft2(event.float())

        # 将RGB和Event的频域表示相乘，然后进行逆变换
        att_fft = frame_fft * event_fft
        att = torch.fft.irfft2(att_fft, s=(H, W))

        # 调整维度: [B, C, H, W] -> [B, H, W, C] 在channel维度归一化
        att = att.permute(0, 2, 3, 1)  # [B, H, W, C]
        att = self.norm1(att)  # 在最后一个维度(C)归一化
        att = att.permute(0, 3, 1, 2)  # [B, C, H, W]

        frame_fused = att * frame
        event_fused = att * event

        # 确保输出是连续的
        frame_fused = frame_fused.contiguous()
        event_fused = event_fused.contiguous()

        return frame_fused, event_fused

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
        # self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        # weight = self.relu2(self.fc2(weight))
        weight = self.fc2(weight)
        weight = self.sigmoid(weight)
        out = x * weight
        return out


class HighOrderChannelInteraction(nn.Module):
    def __init__(self, channels, order=2):
        super().__init__()
        self.order = order
        self.channels = channels

        # 高阶统计建模
        self.high_order_layers = nn.ModuleList()
        for i in range(order):
            self.high_order_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        # x: [B, C, H, W]
        out = x
        for i in range(self.order):
            weight = self.high_order_layers[i](out)
            out = out * weight + x  # 残差连接
        return out

class Counter_attention(nn.Module):
    def __init__(self, inchannels, channel_order=2):
        super(Counter_attention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))
        self.sig = nn.Sigmoid()

        # 提取多尺度特征
        self.mc1 = Multi_Context(inchannels)
        self.mc2 = Multi_Context(inchannels)

        # 高阶通道交互模块
        self.hoc_frame = HighOrderChannelInteraction(inchannels, order=channel_order)
        self.hoc_event = HighOrderChannelInteraction(inchannels, order=channel_order)

        # 自适应权重
        self.ada_w1 = Adaptive_Weight(inchannels)
        self.ada_w2 = Adaptive_Weight(inchannels)

    def forward(self, frame_fre, event_fre, original_frame, original_event):
        """
        frame_fre, event_fre: 频域融合后的特征
        original_frame, original_event: 原始输入特征（用于残差连接）
        """

        # 对事件特征进行多尺度调制（使用帧特征作为指导）
        mc1 = self.mc1(frame_fre)  # 用帧特征生成事件特征的注意力
        pr1 = event_fre * self.sig(mc1)  # 调制事件特征

        # 事件特征的自注意调制
        pr2 = self.conv1(event_fre)
        pr2 = event_fre * self.sig(pr2)

        # 融合 + 残差连接
        out1 = pr1 + pr2 + event_fre + original_event  # 事件分支输出

        # 对帧特征进行多尺度调制（使用事件特征作为指导）
        mc2 = self.mc2(event_fre)  # 用事件特征生成帧特征的注意力
        as1 = frame_fre * self.sig(mc2)  # 调制帧特征

        # 帧特征的自注意调制
        as2 = self.conv2(frame_fre)
        as2 = frame_fre * self.sig(as2)

        # 融合 + 残差连接
        out2 = as1 + as2 + frame_fre + original_frame  # 帧分支输出

        # 新增：高阶通道交互
        out1 = self.hoc_event(out1)
        out2 = self.hoc_frame(out2)

        # 应用自适应权重
        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)

        # 最终拼接
        out = torch.cat([out1, out2], dim=1)

        return out


if __name__ == '__main__':
    batch_size, channels, height, width = 4, 128, 36, 36
    events = torch.randn(batch_size, channels, height, width)
    imgs = torch.randn(batch_size, channels, height, width)

    # 保存原始输入（用于残差连接）
    original_events = events.clone()
    original_imgs = imgs.clone()

    # 一阶：频域交互
    fuse_net = Frequency_Fuse(channels)
    frame_fre, event_fre = fuse_net(imgs, events)

    # 二阶：CDMS空间交互
    cdms_net = Counter_attention(channels)
    fused_features = cdms_net(frame_fre, event_fre, original_imgs, original_events)

    print(f"输入形状: imgs {imgs.shape}, events {events.shape}")
    print(f"频域融合后: frame_fre {frame_fre.shape}, event_fre {event_fre.shape}")
    print(f"最终输出: {fused_features.shape}")  # 应该是 [4, 256, 36, 36]
