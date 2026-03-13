import torch
import torch.nn.functional as F

def AIHD_early(events):
    # 获取输入张量所在的设备
    device_input = events.device

    # 将输入张量移动到 GPU 上（如果可用）
    device_compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    events = events.to(device_compute)

    # 检查输入张量的维度
    if len(events.shape) == 5:
        # 训练时，输入形状为 (images, sequences, channels, height, width)
        images, sequences, channels, height, width = events.shape
        batch_size = images * sequences
        events = events.view(batch_size, channels, height, width)  # 调整为 4D 形状
        output_shape = (images, sequences, channels, height, width)  # 保存原始形状
    elif len(events.shape) == 4:
        # 测试时，输入形状为 (batch_size, channels, height, width)
        batch_size, channels, height, width = events.shape
        output_shape = events.shape  # 保存原始形状
    else:
        raise ValueError('Unsupported input shape. Expected 5D or 4D tensor.')

    # 初始化高密度图
    # high_density_map = torch.zeros((batch_size, channels, height, width), dtype=torch.float32, device=device_compute)

    # 调用 process_batch_image 函数处理 events
    high_density_map = process_batch_image(events)

    # 确保 high_density_map 的形状与处理后的 events 一致
    if high_density_map.shape != (batch_size, channels, height, width):
        raise ValueError("process_batch_image returned an unexpected shape. "
                         f"Expected: {(batch_size, channels, height, width)}, "
                         f"Got: {high_density_map.shape}")

    # 将 high_density_map 调整回原始输入的形状
    high_density_map = high_density_map.view(output_shape)

    # 将结果移动回输入张量所在的设备
    high_density_map = high_density_map.to(device_input)

    return high_density_map


def process_batch_image(batch_images):
    # 对图像进行反相处理
    inverted_images = 255 - batch_images
    # print("inverted_images shape:", inverted_images.shape)
    # (batch_size, channels, height, width)
    batch_size, channels, height, width = batch_images.shape

    # 定义卷积核
    W, H = 7, 7
    kernel = torch.ones((channels, 1, W, H), dtype=torch.float32, device=batch_images.device)
    # print("kernel shape:", kernel.shape)

    # 逐通道进行卷积
    density_map = F.conv2d(inverted_images, kernel, padding=(3, 3), groups=channels)
    # print("density_map shape:", density_map.shape)

    # 归一化密度图
    density_maps_normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8) *255
    # print("density_maps_normalized shape:", density_maps_normalized.shape)

    # 计算自适应阈值
    thresholds = adaptive_threshold_batch(density_maps_normalized)
    thresholds = thresholds.view(-1, 1, 1, 1)  # 形状变为 [13, 1, 1, 1]
    # print("thresholds shape:", thresholds.shape)

    # 根据阈值计算高密度区域
    high_density_images = torch.where(density_maps_normalized < thresholds,
                                      torch.zeros_like(density_maps_normalized),
                                      torch.ones_like(density_maps_normalized))
    # 返回高密度区域的图像
    attention_maps = inverted_images * high_density_images

    # 将其归一化到[0,1]之间
    # attention_maps = attention_maps / 255.0

    return attention_maps

def adaptive_threshold_batch(density_maps):
    # 计算整个批次的自适应阈值。输入density_maps形状为（batch_size, channels, height, width）
    # 输出形状为(batch,)的阈值张量
    batch_size, channels, height, width = density_maps.shape

    # 使用均值滤波计算平均密度
    avg_pools = F.avg_pool2d(density_maps, kernel_size=5, stride=1, padding=2).mean(dim=[2, 3])  # 形状为 [batch_size, channels]

    # 计算直方图的熵
    bins = 64
    histograms = []
    for i in range(batch_size):
        for j in range(channels):
            hist = torch.histc(density_maps[i, j].flatten(), bins=bins, min=0, max=255)
            histograms.append(hist)
    histograms = torch.stack(histograms).view(batch_size, channels, bins)  # 形状为 [batch_size, channels, bins]

    histograms = histograms / histograms.sum(dim=2, keepdim=True)  # 归一化直方图
    entropies = -torch.sum(histograms * torch.log(histograms + 1e-8), dim=2)  # 形状为 [batch_size, channels]

    # 计算自适应阈值
    thresholds = avg_pools + entropies * 30
    return thresholds.mean(dim=1)  # 返回形状为 [batch_size]

