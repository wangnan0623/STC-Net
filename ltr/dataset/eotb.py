import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import string
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings

# 定义类别字典，包含三大类
cls = {'animal':['dove','bear','elephant','cow','giraffe','dog','turtle','whale'],
'vehicle':['toy_car','airplane','fighter','truck','ship','tank','suv','bike'],
'object':['ball','star','cup','box','bottle','tower']}

class EOTB(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None):
        """
        args:
            root - path to the FE108 dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
        """
        root = env_settings().eotb_dir if root is None else root
        super().__init__('EOTB', root, image_loader)

        self.sequence_list = self._get_sequence_list()
        # 根据split的类型读取相应的划分文件，并从中读取序列名称，构造列表seq_name
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'eotb_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'eotb_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
        else:
            seq_names = self.sequence_list
        # 根据获得的序列名称更新序列列表
        self.sequence_list = [i for i in seq_names]
        # 加载元信息
        self.sequence_meta_info = self._load_meta_info()
        # 构建每个类别对应的序列索引
        self.seq_per_class = self._build_seq_per_class()
        # 获取并排序类别列表
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _build_seq_per_class(self):
        # 将序列按照类别进行分类
        # {
        #     'car': [0, 2],  # 'sequence_1' 和 'sequence_3' 属于 'car'
        #     'pedestrian': [1]  # 'sequence_2' 属于 'pedestrian'
        # }

        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            # 检查object_class是否已经存在于seq_per_class字典中
            # 如果存在，将当前的序列索引i添加到相应的列表中
            # 如果不存在，则初始化一个新列表，将当前索引i包含在内
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def _get_sequence_list(self):
        seq_list = os.listdir(self.root)
        return seq_list

    def _load_meta_info(self):
        # 读取每个序列的元数据，构建字典，其键为sequence_list中每个序列名称
        # 形如：
        # {
        #     'airplane': {'object_class_name': 'airplane', 'motion_class': None,'major_class': None,'root_class': None,'motion_adverb': None},
        #     'ball333': {'object_class_name': 'ball', 'motion_class': None,'major_class': None,'root_class': None,'motion_adverb': None},
        #     'dog_mul': {'object_class_name': 'dog', 'motion_class': None,'major_class': None,'root_class': None,'motion_adverb': None}
        # }

        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'eotb'

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_meta(self, seq_path):
        # 获取序列的类别，并构建字典，只传入类别，其他信息暂时为None，将字典作为返回值，其中保存了序列的元数据
        obj_class = self._get_class(seq_path)
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_frame_path(self, seq_path, frame_id):
        if '20fps' in seq_path:
            beishu = 12
        elif '15fps' in seq_path:
            beishu = 16
        elif '10fps' in seq_path:
            beishu = 24
        else:
            beishu = 6
        gray_id = frame_id // beishu + 1
        img_path = os.path.join(seq_path, 'img', '{:04}.jpg'.format(gray_id))

        return img_path

    def _get_frame(self, seq_path, frame_id):
        img_path = self._get_frame_path(seq_path, frame_id)
        # img_path = os.path.join(seq_path, 'img', '{:04}.jpg'.format(frame_id+1))
        return self.image_loader(img_path)

    def _get_event(self, seq_path, frame_id):
        pos_path = os.path.join(seq_path, 'accumulate_events', '{:04}.jpg'.format(frame_id+1))
        return self.image_loader(pos_path)

    def _get_frames(self, seq_id):
        path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def _get_class(self, seq_path):
        # 通过从路径字符串seq_path中去掉后面的，只保留前面的信息，并将其作为类别
        raw_class = seq_path.split('/')[-1].rstrip(string.digits).split('_')[0]
        return raw_class

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)  # 验证边界框的有效性
        visible = valid.clone().byte()  # 将有效的边界框信息克隆到visible数组中，并将其转换为字节类型

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        event_list = [self._get_event(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return event_list, frame_list, anno_frames, obj_meta
