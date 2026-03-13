import os
import torch.optim as optim
from ltr.dataset import EOTB, VisEvent, FE108
from ltr.data import processing, sampler, LTRLoader
# from ltr.models.tracking import dimpnet
from ltr.models.tracking import aihdnet
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU


def run(settings):
    settings.description = 'Default train settings for ahnet with ResNet18 as backbone.'
    settings.batch_size = 20  # 默认为32
    settings.num_workers = 4  # 设置用于数据加载的工作线程数量，以加快数据读取速度  默认为32  取4
    settings.multi_gpu = True  # 是否使用多GPU训练
    settings.print_interval = 1  # 设置打印输出的间隔为1，即每个训练周期都打印一次状态信息
    settings.normalize_mean = [0.485, 0.456, 0.406]  # 设置图像标准化的均值
    settings.normalize_std = [0.229, 0.224, 0.225]  # 设置图像标准化的标准差
    settings.search_area_factor = 5.0  # 设置搜索区域因子，用于扩大或缩小跟踪目标的搜索区域
    settings.output_sigma_factor = 1/4  # 设置输出的sigma因子，用于调整输出的标准差大小
    settings.target_filter_sz = 4  # 目标滤波器大小
    settings.feature_sz = 18  # 特征大小
    settings.output_sz = settings.feature_sz * 16  # 输出大小，通常与特征大小成比例
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}  # 设置中心抖动因子，在训练和测试时有不同的值，以增加数据的多样性
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}  # 设置缩放抖动因子
    settings.hinge_threshold = 0.05  # 合页阈值，通常用于损失函数计算
    settings.print_stats = ['Loss/total', 'Loss/bb_ce', 'ClfTrain/clf_ce']  # 设置要打印的统计信息列表，包括总损失、边界框分类损失、分类训练损失
    # 新增测试间隔设置
    settings.test_interval = 1  # 每5个epoch测试一次

    # # Train datasets FE108
    # train_dataset_path = os.path.join(settings.env.FE108_dir, 'train')
    # eotb_train = FE108(train_dataset_path)
    #
    # # Validation datasets FE108
    # test_dataset_path = os.path.join(settings.env.FE108_dir, 'test')
    # eotb_test = FE108(test_dataset_path)

    # Train datasets VisEvent
    train_dataset_path = os.path.join(settings.env.VisEvent_dir, 'train_subset')
    eotb_train = VisEvent(train_dataset_path)

    # Validation datasets VisEvent
    test_dataset_path = os.path.join(settings.env.VisEvent_dir, 'test_subset')
    eotb_test = VisEvent(test_dataset_path)


    # Data transform 定义联合变换，将图像转为灰度图，概率为5%
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # 定义训练时的变换，包括将图像转换为张量并施加抖动，以及进行标准化
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # 定义验证时的变换，将图像转为张量并进行标准化
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor  # 计算输出的sigma，用于标定输出的标准差
    # 设置提议参数，包括每帧的边界框数量、真实值的sigma和提议的sigma
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    # 设置标签参数，包括特征大小、sigma因子和核大小
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    # 设置标签密度参数，包括特征大小、sigma因子、核大小以及是否进行归一化
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz, 'normalize': True}

    # 初始化训练数据处理模块
    data_processing_train = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    # 初始化验证数据处理模块，配置与训练相似
    data_processing_val = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      label_density_params=label_density_params,
                                                      transform=transform_val,
                                                      joint_transform=transform_joint)

    # Train sampler and loader
    # 使用sample.DiMPSampler从训练数据集etob_train中根据指定参数生成训练样本dataset_train
    # loader_train负责从训练样本sataset_train中以特定方式迭代产生一个个batch的样本集合
    dataset_train = sampler.DiMPSampler([eotb_train], [1],
                                        samples_per_epoch=26000, max_gap=200, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    # 验证集与训练集类似（采样数量从5000修改为10000）
    dataset_val = sampler.DiMPSampler([eotb_test], [1], samples_per_epoch=10000, max_gap=200,
                                      num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=10, stack_dim=1)

    # Create network and actor
    # # 创建DIMP网络实例，使用特定的超参数配置，如滤波器大小，是否使用预训练模型等
    # net = dimpnet.klcedimpnet18(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
    #                         clf_feat_norm=True, final_conv=True, optim_init_step=1.0, optim_init_reg=0.05, optim_min_reg=0.05,
    #                         gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True, init_initializer='zero')

    # 创建aihd网络（backbone_pretrained设为了False）
    pretrained_path = "/home/wangnan/project/AFNet-main/logs/FE108_LIF/checkpoints/ltr/ahnet/ahnet/AIHDnet_ep0005.pth.tar"
    net = aihdnet.klcedimpnet18(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                                clf_feat_norm=True, final_conv=True, optim_init_step=1.0, optim_init_reg=0.05,
                                optim_min_reg=0.05,
                                gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True,
                                init_initializer='zero', pretrained_path=None)

    ############################显式冻结LIF参数 - 添加这部分##############################
    # print("Freezing LIF parameters...")
    # for name, param in net.named_parameters():
    #     if 'LIF' in name:
    #         param.requires_grad = False
    #         print(f"Frozen: {name}")
    #
    # # 验证冻结效果
    # print("\n=== Parameter Status ===")
    # lif_count = 0
    # trainable_count = 0
    # for name, param in net.named_parameters():
    #     if 'LIF' in name:
    #         lif_count += 1
    #         if param.requires_grad:
    #             print(f"WARNING: {name} is still trainable!")
    #     elif param.requires_grad:
    #         trainable_count += 1
    #
    # print(f"Frozen LIF parameters: {lif_count}")
    # print(f"Trainable parameters: {trainable_count}")
    #####################################################################

    # Wrap the network for multi GPU training
    # 如果启用了多GPU训练，则将网络包装为multi_gpu模式
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    # 定义训练目标，使用KL回归损失函数来计算边界框和分类损失
    objective = {'bb_ce': klreg_losses.KLRegression(), 'clf_ce': klreg_losses.KLRegressionGrid()}

    # 设置各个损失的权重，以便在计算总体损失时使用
    loss_weight = {'bb_ce': 0.0025, 'clf_ce': 0.25, 'clf_ce_init': 0.25, 'clf_ce_iter': 1.0}

    # 创建actor实例，负责管理网络、损失目标和损失权重
    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer 创建Adam优化器(全设为1e-4)
    optimizer = optim.Adam([{'params': actor.net.classifier.parameters(), 'lr': 5e-5},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.LIF.parameters(), 'lr': 1e-5},
                            {'params': actor.net.fusion.parameters(), 'lr': 5e-5}])


    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    # 定义学习率调度器，使用余弦退火调度器来调整学习率，每50个epoch学习率重置1次
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 创建LTRTrainer实例，负责训练和验证过程
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # 启动训练过程，设置训练epoch为60，同时支持加载最新模型和失败安全机制
    trainer.train(60, load_latest=True, fail_safe=True)
