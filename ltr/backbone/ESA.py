import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class EventSparseAttention(nn.Module):
    def __init__(self, kernel_size=7, top_k_ratio=0.1, num_iterations=1):
        super(EventSparseAttention, self).__init__()
        self.top_k_ratio = top_k_ratio
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size

    def forward(self, x):
        # 检查输入张量的维度
        if len(x.shape) == 5:
            # 训练时，输入形状为(images, sequences, channels, height, width)
            images, sequences, channels, height, width = x.shape
            batch_size = images * sequences
            x = x.view(batch_size, channels, height, width)  # 调整为4D形状
            output_shape = (images, sequences, channels, height, width)  # 保存原始形状
        elif len(x.shape) == 4:
            # 测试时，输入形状为(batch_size, channels, height, width)
            batch_size, channels, height, width = x.shape
            output_shape = x.shape
        else:
            raise ValueError('Unsupported input shape. Expected 5D or 4D tensor.')

        attention_matrix = x
        for _ in range(self.num_iterations):
            scores = self.calculate_score(attention_matrix)
            k = int(self.top_k_ratio * scores.numel())
            top_k_indices = torch.topk(scores.view(-1), k, dim=0).indices
            mask = torch.zeros_like(scores.view(-1))
            mask[top_k_indices] = 1
            mask = mask.view_as(scores)
            attention_matrix = attention_matrix * mask

        # 恢复原始形状
        if len(output_shape) == 5:
            attention_matrix = attention_matrix.view(*output_shape)

        return attention_matrix

    def calculate_score(self, attention_matrix):
        # 计算局部最大值
        max_val = self.max_pool(attention_matrix)
        # 计算局部平均值
        avg_val = self.avg_pool(attention_matrix)
        # 计算局部对比度
        contrast = torch.abs(max_val - avg_val)
        return contrast

    def max_pool(self, x):
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=3)

    def avg_pool(self, x):
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=3)


# 示例使用
if __name__ == "__main__":
    # 创建一个模型实例
    model = EventSparseAttention(kernel_size=7, top_k_ratio=0.1, num_iterations=1)

    # 创建一个模拟的输入张量，形状为 (images, sequences, channels, height, width)
    images, sequences, channels, height, width = 4, 2, 3, 64, 64
    input_tensor_5d = torch.randn(images, sequences, channels, height, width)

    # 创建一个模拟的输入张量，形状为 (batch_size, channels, height, width)
    batch_size, channels, height, width = 8, 3, 64, 64
    input_tensor_4d = torch.randn(batch_size, channels, height, width)

    # 应用模型
    output_5d = model(input_tensor_5d)
    output_4d = model(input_tensor_4d)

    # 打印输出形状
    print("Output shape (images, sequences, channels, height, width):", output_5d.shape)
    print("Output shape (batch_size, channels, height, width):", output_4d.shape)