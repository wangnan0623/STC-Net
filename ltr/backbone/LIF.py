import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import cv2
from ltr.models.backbone import attention_module
from ltr.models.backbone.attention_module import TemporalAttention


def recover_fast_inputs(input_array, spk_output_array, recovery_neighborhood=5):
    # 根据恢复的邻域设置，定义全1的卷积核
    kernel = np.ones((recovery_neighborhood, recovery_neighborhood), np.uint8)
    # 对脉冲输出进行膨胀运算，将孤立的脉冲点扩展为邻域范围（覆盖可能的输入事件区域）
    dilated_speedy_img = cv2.dilate(np.array(spk_output_array), kernel, iterations = 1)
    # 形态学运算，先膨胀，后腐蚀，使掩模更连续
    closing = cv2.morphologyEx(dilated_speedy_img, cv2.MORPH_CLOSE, kernel)
    # 创建掩模（仅保留input_array和closing同时非零的位置），用掩模过滤原始输入，保留可能触发脉冲的事件区域
    masked_input = np.array(np.logical_and(input_array,closing))*input_array
    return masked_input


# 更高效的批量处理版本（使用向量化操作）
class EfficientAdaptiveThreshold(nn.Module):
    def __init__(self, num_bins=256):
        super(EfficientAdaptiveThreshold, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()
        self.num_bins = num_bins
        self.bin_edges = torch.linspace(0, 255, num_bins + 1, requires_grad=False)

    def compute_entropy_batch(self, xn_normalized):
        """批量计算熵 - 向量化版本"""
        batch_size, n_elements = xn_normalized.shape

        # 计算直方图
        bin_indices = (xn_normalized / (256 / self.num_bins)).long().clamp(0, self.num_bins - 1)

        # 使用one-hot编码和求和来计算直方图
        one_hot = torch.zeros(batch_size, self.num_bins, device=xn_normalized.device)
        one_hot.scatter_add_(1, bin_indices, torch.ones_like(xn_normalized))

        # 归一化概率
        probs = one_hot / (one_hot.sum(dim=1, keepdim=True) + 1e-8)

        # 计算熵
        nonzero_mask = probs > 0
        entropy = -torch.where(nonzero_mask, probs * torch.log(probs + 1e-8), torch.zeros_like(probs))
        entropy = entropy.sum(dim=1) / nonzero_mask.sum(dim=1).clamp(min=1)

        return entropy

    def forward(self, x):
        """
        向量化版本的AdaptiveThreshold
        Args:
            x: 输入张量 [batch, C, H, W]
        Returns:
            阈值张量 [batch]
        """
        batch_size, C, H, W = x.size()

        # 计算每个样本的特征
        xn = x * self.avg_pool(x)  # [batch, C, H, W]
        xn_mean = xn.mean(dim=1, keepdim=True)  # [batch, 1, H, W]

        # 重塑以便批量处理
        xn_flat = xn_mean.view(batch_size, -1)  # [batch, H*W]

        # 计算每个样本的归一化值
        min_vals = xn_flat.min(dim=1)[0]  # [batch]
        max_vals = xn_flat.max(dim=1)[0]  # [batch]

        # 避免除零
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # 归一化到0-255
        xn_normalized = (xn_flat - min_vals.unsqueeze(1)) / range_vals.unsqueeze(1) * 255
        xn_normalized = xn_normalized.clamp(0, 255)

        # 批量计算熵
        entro_tensor = self.compute_entropy_batch(xn_normalized)  # [batch]

        # 计算sigmoid部分
        sig_part = self.sig(xn_mean)  # [batch, 1, H, W]
        sig_part = sig_part.mean(dim=[1, 2, 3])  # [batch]

        return sig_part + entro_tensor * 10

class AdaptiveLIF(nn.Module):
    def __init__(self, in_channels=3):
        super(AdaptiveLIF, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 手动设置卷积核权重
        with torch.no_grad():
            self.conv1.weight.data.fill_(0.15)
            for c in range(in_channels):
                self.conv1.weight.data[c, c, 1, 1] = 0.2  # 对角线中心权重

        self.adaptive_threshold = EfficientAdaptiveThreshold()
        self.LIF = snn.Leaky(beta=0.3, reset_mechanism="subtract")
        self.temporal_attention = TemporalAttention(time_windows=4, reduction=2)

    def forward(self, x):
        """
          Args:
              x: 输入张量 [T, frame_num * batch, C, H, W]
          Returns:
              脉冲输出 [batch, C, H, W]
          """

        T, frame_num, C, H, W = x.shape

        # # 初始化膜电位
        # mem = self.LIF.init_leaky()
        #
        # # 在时间维度串行输入
        # spk_output = []
        # for t in range(T):
        #     # 获取当前时间步所有batch的数据
        #     x_t = x[:, t]  # [B, C, H, W]
        #     x_t = self.conv1(x_t)  # [B, C, H, W]
        #     # 获取当前时间步所有batch的阈值
        #     thresholds = self.adaptive_threshold(x_t)  # [B]
        #     mean_thresholds = thresholds.mean()
        #
        #     self.LIF.threshold = mean_thresholds
        #     spk, mem = self.LIF(x_t, mem=mem)  # spk和 mem都是[batch_size, C, H, W]
        #     spk_output.append(spk.unsqueeze(1))  # 添加时间维度 [batch_size, 1, C, H, W]
        #
        # # 将时间步上的输出拼接 [batch_size, T, C, H, W]
        # all_spk_outputs = torch.cat(spk_output, dim=1)

        # 应用时间注意力
        # all_spk_outputs = self.temporal_attention(all_spk_outputs )
        all_spk_outputs = self.temporal_attention(x)


        # 对时间步取平均 [batch, C, H, W]
        # final_output = torch.mean(all_spk_outputs, dim=1)

        return all_spk_outputs


# 测试代码
def test_model():
    # 创建测试数据
    batch_size, T, C, H, W = 2, 4, 3, 32, 32
    test_input = torch.randn(batch_size, T, C, H, W)

    # 初始化模型
    model = AdaptiveLIF(in_channels=C)

    # 前向传播测试
    try:
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print("模型测试通过!")
        return True
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False


if __name__ == "__main__":
    test_model()
