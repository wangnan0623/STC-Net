import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from ltr.models.backbone import attention_module
from ltr.models.backbone import frequency_feature


class FUSION(nn.Module):
    def __init__(self):
        super(FUSION, self).__init__()

        # 提取频域特征
        self.freq_low = frequency_feature.Frequency_Fuse(128)
        self.freq_high = frequency_feature.Frequency_Fuse(256)

        # CDMS模块
        # self.counter_atten_low = attention_module.Counter_attention(128)
        # self.counter_atten_high = attention_module.Counter_attention(256)

        self.counter_att_low = frequency_feature.Counter_attention(128)
        self.counter_att_high = frequency_feature.Counter_attention(256)

    def forward(self, frame_features_low, event_features_low, frame_features_high, event_features_high):
        # 原本单帧时,输入数据:
        # input: frame_feature_low, event_feature_low [24, 128, 36, 36]
        # input: frame_feature_high, event_feature_high [24, 256, 18, 18]
        # 测试时，初始化，low [13, 128, 36, 36]

        # 直接拼接融合
        # feature_low = torch.cat([frame_features_low, event_features_low], dim=1)
        # feature_high = torch.cat([frame_features_high, event_features_high], dim=1)


        # 一阶：频域交互
        frame_fre_low, event_fre_low = self.freq_low(frame_features_low, event_features_low)
        frame_fre_high, event_fre_high = self.freq_high(frame_features_high, event_features_high)
        #
        # feature_low = torch.cat((frame_fre_low, event_fre_low), dim=1)
        # feature_high = torch.cat((frame_fre_high, event_fre_high), dim=1)

        # 二阶：CDMS空间交互
        feature_low = self.counter_atten_low(frame_features_low, event_features_low)
        feature_high = self.counter_atten_high(frame_features_high, event_features_high)

        return feature_low, feature_high

