import os
import glob
import torch
import traceback
from ltr.admin import loading, multigpu
import subprocess
import re



class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function.
    基础训练器。定义了如何更新设置、保存和载入checkpoint。具体如何训练需要由子类重新定义train_epoch方法"""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)


    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:  # 如果提供了新的settings，则更新内部的self.settings
            self.settings = settings

        if self.settings.env.workspace_dir is not None:  # 如果工作目录不为空
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)  # 将其展开为绝对路径
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')  # 创建checkpoints目录
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.是否从最新的周期恢复训练
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.在发生错误时是否自动重新启动训练
        """

        epoch = -1
        num_tries = 10  # 设置最大重试次数为10
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()

                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()  # 调用train_epoch方法执行每个周期的训练

                    if self.lr_scheduler is not None:  # 如果有学习率调度器，则在每个周期结束时调用其step方法调整学习率
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:  # 如果checkpoint目录存在，则保存当前模型
                        self.save_checkpoint()

            except:  # 处理异常
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')


    def train_epoch(self):
        # 定义一个抽象方法train_epoch，要求子类必须实现该方法以执行具体的训练逻辑
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        # 从actor中获取网络类型
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        # 获取actor和net的类型名称
        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),  # 网络的状态字典
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),  # 优化器的状态字典
            'stats': self.stats,
            'settings': self.settings
        }


        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file 临时保存
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. 将临时文件重命名为正式的checkpoint文件
        # os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        # 确定要加载的字段和要忽略的字段
        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info 设置网络信息
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True
