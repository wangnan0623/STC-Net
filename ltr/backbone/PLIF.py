import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, overload
from spikingjelly.clock_driven import functional, layer, surrogate
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms
import math

class BALIFBetaAdaptor(nn.Module):
    """动态膜时间常数适配器"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.channel_attention = nn.Sequential(
            # 先用1x1卷积降维
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # 再用1x1卷积恢复维度并生成权重
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出在[0,1]范围内
        )

    #     # 初始化参数
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     # 初始化最后一层卷积的bias，使初始beta≈0.9
    #     if hasattr(self.channel_attention[3], 'bias') and self.channel_attention[3].bias is not None:
    #         nn.init.constant_(self.channel_attention[3].bias, 0.9)  # sigmoid(0.9) ≈ 0.71

    def forward(self, x):
        """
        x: (B, C, H, W) 输入事件帧
        返回: (B, C, 1, 1) 每个通道的beta值，范围[0.5, 0.95]
        """
        B, C, H, W = x.shape

        # BALIF双路径特征提取
        max_pool = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        pooled_features = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2*C, 1, 1)

        # 通过通道注意力生成beta权重
        beta_weights = self.channel_attention(pooled_features)  # (B, C, 1, 1)

        # 将权重映射到[0.5, 0.95]范围，避免出现极端值
        beta = 0.5 + 0.45 * beta_weights  # (B, C, 1, 1)

        return beta

class LightweightEventAdaptor(nn.Module):
    """根据输入事件的特征动态调整神经元阈值"""
    def __init__(self, base_threshold=1.0, buffer_size=4):
        super().__init__()
        self.base_threshold = base_threshold
        self.buffer_size = buffer_size

        # 三个可学习的权重参数
        self.density_weight = nn.Parameter(torch.tensor(1.0))  # 事件密度权重
        self.temporal_weight = nn.Parameter(torch.tensor(0.8))  # 时序变化权重
        self.motion_weight = nn.Parameter(torch.tensor(0.5))  # 运动模式权重

    def forward(self, current_events, previous_events=None):
        """
        current_events: (B, C, H, W) 当前事件帧
        previous_events: 之前的事件帧（用于时序分析）
        """
        B, C, H, W = current_events.shape

        # 计算每个batch中事件的平均密度（密度高，阈值就要提高），每个样本对应一个密度值
        density = torch.mean(current_events.view(B, -1), dim=1)  # (B,)

        # 计算当前帧与前一帧的差异程度（变化大，阈值就要提高）
        if previous_events is not None:
            # 计算与前一帧的差异
            temporal_change = torch.mean(torch.abs(current_events - previous_events).view(B, -1), dim=1)
        else:
            temporal_change = torch.zeros(B, device=current_events.device)

        # 计算局部3*3邻域内的方差，衡量事件的聚焦程度（方差大、事件分布不均匀，阈值就要提高）
        unfolded = F.unfold(current_events, kernel_size=3, padding=1)  # (B, 9, H*W)
        local_variance = torch.var(unfolded, dim=1)  # (B, H*W)
        motion_pattern = torch.mean(local_variance, dim=1)  # (B,)

        # 加权融合三个特征
        threshold_adjust = (self.density_weight * density +
                            self.temporal_weight * temporal_change +
                            self.motion_weight * motion_pattern)

        # 应用sigmoid约束调整范围
        threshold_adjust = torch.sigmoid(threshold_adjust) * 2.0  # 限制在[0, 2]范围内

        # 最终阈值 = 基础阈值 + 调整量
        return self.base_threshold + threshold_adjust.reshape(B, 1, 1, 1)


class AdaptiveLIF(nn.Module):
    def __init__(self, in_channels=1, base_threshold=1.0, reduction_ratio=16,
                 v_reset=0.0, detach_reset=False, surrogate_function=None):
        super().__init__()
        self.in_channels = in_channels
        self.base_threshold = base_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset

        # 替代梯度函数
        if surrogate_function is None:
            self.surrogate_function = self.atan_surrogate
        else:
            self.surrogate_function = surrogate_function

        # 自适应阈值模块
        self.threshold_adaptor = LightweightEventAdaptor(base_threshold)
        # 膜时间常数
        self.beta_adaptor = BALIFBetaAdaptor(in_channels, reduction_ratio)

        # 定义3x3邻域卷积，并初始化权重
        self.neighbor_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                     padding=1, bias=False, groups=in_channels)
        self._initialize_weights(self.neighbor_conv)

        # 膜电势状态
        self.v = None

    def _initialize_weights(self, x):
        """正确初始化权重，支持多通道"""
        with torch.no_grad():
            # 创建正确的权重形状: [out_channels, in_channels/groups, 3, 3]
            # 对于groups=in_channels的情况，应该是 [in_channels, 1, 3, 3]
            base_kernel = torch.tensor([
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1]
            ]).unsqueeze(0).unsqueeze(0)  # 形状: [1, 1, 3, 3]

            # 复制到所有通道
            weight = base_kernel.repeat(self.in_channels, 1, 1, 1)
            x.weight.data = weight

    def atan_surrogate(self, x, alpha=2.0):
        return torch.atan(alpha * x) * 0.5 + 0.5

    def neuronal_charge(self, x, beta):
        """使用动态beta的充电过程"""
        # 更新膜电势: V[t] = β * V[t-1] + (1 - β) * X[t]
        decay_component = beta * self.v
        input_component = (1.0 - beta) * x
        self.v = decay_component + input_component

    def neuronal_fire(self, adaptive_threshold):
        return self.surrogate_function(self.v - adaptive_threshold)

    def neuronal_reset(self, spike, adaptive_threshold):
        """需要传入adaptive_threshold"""
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # 软重置
            self.v = self.v - spike_d * adaptive_threshold
        else:
            # 硬重置
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def initialize_states(self, batch_size, spatial_size, device):
        if self.v_reset is None:
            self.v = torch.zeros(batch_size, 1, *spatial_size, device=device)
        else:
            self.v = torch.full((batch_size, 1, *spatial_size), self.v_reset, device=device)

    def forward(self, events):
        T, B, C, H, W = events.shape

        if self.v is None or self.v.shape[0] != B:
            self.initialize_states(B, (H, W), device=events.device)

        processed_frames = []
        previous_frame = None

        for t in range(T):
            current_frame = events[t]

            # 应用邻域加权
            weighted_input = self.neighbor_conv(current_frame)

            # 根据输入动态计算膜时间常数
            dynamic_beta = self.beta_adaptor(current_frame)  # (B, C, 1, 1)

            # 计算自适应阈值
            adaptive_threshold = self.threshold_adaptor(current_frame, previous_frame)

            # 使用 no_grad 来更新膜电势状态，避免梯度问题
            with torch.no_grad():
            # PLIF三步
                self.neuronal_charge(weighted_input, dynamic_beta)  # 充电
                spike_output = self.neuronal_fire(adaptive_threshold)  # 发放（传入阈值）
                self.neuronal_reset(spike_output, adaptive_threshold)  # 重置

            # 恢复原始输入事件
            output_events = current_frame * spike_output

            processed_frames.append(output_events)
            previous_frame = current_frame
        # 输出与输入形状相同，[T,B,C,H,W]
        out = torch.stack(processed_frames, dim=0)

        return out

    def extra_repr(self):
        return f'beta={self.beta.item():.3f}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'


def test_adaptive_lif():
    """测试AdaptiveLIF模块"""
    print("=== AdaptiveLIF 测试用例 ===")

    # 设置随机种子保证可重复性
    torch.manual_seed(42)

    # 创建测试数据：模拟4个时间步，batch=2，1个通道，8x8分辨率
    # 数据格式: (T, B, C, H, W) = (4, 2, 1, 8, 8)
    T, B, C, H, W = 4, 6, 3, 288, 288
    test_events = torch.randn(T, B, C, H, W)

    print(f"输入数据形状: {test_events.shape}")
    print(f"输入数据范围: [{test_events.min():.3f}, {test_events.max():.3f}]")

    # 创建AdaptiveLIF模块
    adaptive_lif = AdaptiveLIF(in_channels=3, base_threshold=1)

    print(f"\n模块参数:")
    print(f"基础阈值: {adaptive_lif.base_threshold}")
    # print(f"初始beta值: {adaptive_lif.beta_adaptor.data}")
    print(f"卷积核形状: {adaptive_lif.neighbor_conv.weight.shape}")
    print(f"密度权重: {adaptive_lif.threshold_adaptor.density_weight.data}")
    print(f"时序权重: {adaptive_lif.threshold_adaptor.temporal_weight.data}")
    print(f"运动权重: {adaptive_lif.threshold_adaptor.motion_weight.data}")

    # 前向传播
    with torch.no_grad():
        output_events = adaptive_lif(test_events)

    print(f"\n输出数据形状: {output_events.shape}")


if __name__ == "__main__":
    test_adaptive_lif()
