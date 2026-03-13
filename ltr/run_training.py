import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
print("PyTorch sees device count:", torch.cuda.device_count())
print("Physical GPU indices (via CUDA_VISIBLE_DEVICES):", os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))
print("Current device id:", torch.cuda.current_device())
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
import ltr.admin.settings as ws_settings
# 暂时关闭警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


env_path = os.path.join(os.path.dirname(__file__), '..')  # 获取当前脚本的目录，并构造上级目录
if env_path not in sys.path:  # 检查构造的路径是否在系统路径中，如果不在，则添加进去
    sys.path.append(env_path)


def run_training(train_module, train_name, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    # 将opencv的线程数设置为0，以避免与其他库发生冲突
    cv.setNumThreads(0)

    # 根据传入参数设置是否启用cuDNN基准模式，以优化卷积计算速度
    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    # 导入指定的训练模块，并从导入的模块中获取run函数，将其赋给expr_func
    # 以afnet为例，导入ltr.train_setting.afnet.afnet文件，其中定义了run函数
    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    # 执行expr_func，并将配置好的settings作为参数传递
    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, default='ahnet', help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, default='ahnet', help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()

    # 调用上面定义的run_training函数
    run_training(args.train_module, args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':  # 确保只有直接运行该脚本时才会执行以下代码（而不是被导入）
    # 设置多进程启动方法为spawn
    multiprocessing.set_start_method('spawn', force=True)
    main()
