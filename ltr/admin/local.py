class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wangnan/project/AFNet-main/logs/test'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.eotb_dir = ''
        self.FE108_dir = '/home/Data/FE240'
        self.VisEvent_dir = '/home/Data/VisEvent'
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
