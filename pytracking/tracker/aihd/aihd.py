from pytracking.tracker.dimp import DiMP
from ltr.models.backbone.aihd_early import AIHD_early
from ltr.models.tracking import aihdnet
from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation

class AIHD(DiMP):
    """AIHD model class, inheriting from DiMP and modifying the feature fusion part."""

    # def __init__(self, params):
    #     super().__init__(params)

    def extract_backbone_features(self, im: torch.Tensor, event, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))

        event_patches = []
        for i in range(len(event)):
            event_patch, patch_coords = sample_patch_multiscale(event[i], pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
            event_patches.append(event_patch)

        # im_patches和event_patch在CPU上
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)

            # 提取事件特征
            event_patches = torch.stack(event_patches, dim=0)
            event_feat_low, event_feat_high = self.net.extract_backbone_event(event_patches)
            #
            # # event_feat = self.net.extract_backbone(event_patches)
            #
            # fusion融合
            train_low, train_high = self.net.fusion(backbone_feat['layer2'], event_feat_low,
                                                    backbone_feat['layer3'], event_feat_high)
            #
            backbone_feat['layer2'] = train_low
            backbone_feat['layer3'] = train_high

        return backbone_feat, patch_coords, im_patches

    def generate_init_samples(self, im: torch.Tensor, event_im) -> TensorList:
        """Perform data augmentation to generate initial training samples.
        im: (threads, C, H, W)
        event_im: {list=4} [(threads, C, H, W)]
        """

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
        event_patches = []
        for i in range(len(event_im)):
            event_patch = sample_patch_transformed(event_im[i], self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
            event_patches.append(event_patch)

        # Extract initial backbone features
        with torch.no_grad():
            # event_patches: {list:4} [Tensor:(13,3,288,288)]
            # im_patches: Tensor:(13,3,288,288)
            init_backbone_feat = self.net.extract_backbone(im_patches)

            # 调整输入事件数据的格式
            event_patches = torch.stack(event_patches, dim=0)  # (4, 13, 3, 288, 288)
            event_init_feat_low, event_init_feat_high = self.net.extract_backbone_event(event_patches)
            #
            # # event_init_feat = self.net.extract_backbone(event_patches)
            # #
            # fusion融合
            train_low_init, train_high_init = self.net.fusion(init_backbone_feat['layer2'], event_init_feat_low,
                                                              init_backbone_feat['layer3'], event_init_feat_high)
            #
            init_backbone_feat['layer2'] = train_low_init
            init_backbone_feat['layer3'] = train_high_init

        return init_backbone_feat




