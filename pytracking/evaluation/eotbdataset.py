import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text

class EOTBDataset(BaseDataset):
    """ OTB-2015 dataset

    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf

    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.eotb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']  # 事件帧的路径
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])
        # print('base_path:{}', self.base_path)
        # print('anno_path:{}', anno_path)


        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'eotb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [  {'anno_path': 'airplane_mul222/groundtruth_rect.txt',
                                'endFrame': 2051,
                                'ext': 'jpg',
                                'name': 'airplane_mul222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'airplane_mul222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'bike_low/groundtruth_rect.txt',
                                 'endFrame': 1290,
                                 'ext': 'jpg',
                                 'name': 'bike_low',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'bike_low/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'bike222/groundtruth_rect.txt',
                                 'endFrame': 1899,
                                 'ext': 'jpg',
                                 'name': 'bike222',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'bike222/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'bike333/groundtruth_rect.txt',
                                 'endFrame': 2001,
                                 'ext': 'jpg',
                                 'name': 'bike333',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'bike333/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'bottle_mul222/groundtruth_rect.txt',
                                 'endFrame': 1101,
                                 'ext': 'jpg',
                                 'name': 'bottle_mul222',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'bottle_mul222/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'box_hdr/groundtruth_rect.txt',
                                'endFrame': 1948,
                                'ext': 'jpg',
                                'name': 'box_hdr',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'box_hdr/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'box_low/groundtruth_rect.txt',
                                'endFrame': 2084,
                                'ext': 'jpg',
                                'name': 'box_low',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'box_low/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'cow_mul222/groundtruth_rect.txt',
                                 'endFrame': 2231,
                                 'ext': 'jpg',
                                 'name': 'cow_mul222',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'cow_mul222/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'cup_low/groundtruth_rect.txt',
                                 'endFrame': 1933,
                                 'ext': 'jpg',
                                 'name': 'cup_low',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'cup_low/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'cup222/groundtruth_rect.txt',
                                'endFrame': 2010,
                                'ext': 'jpg',
                                'name': 'cup222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'cup222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'dog/groundtruth_rect.txt',
                                'endFrame': 642,
                                'ext': 'jpg',
                                'name': 'dog',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'dog/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'dog_motion/groundtruth_rect.txt',
                                'endFrame': 2788,
                                'ext': 'jpg',
                                'name': 'dog_motion',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'dog_motion/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'dove_motion/groundtruth_rect.txt',
                                 'endFrame': 2202,
                                 'ext': 'jpg',
                                 'name': 'dove_motion',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'dove_motion/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'dove_mul/groundtruth_rect.txt',
                                 'endFrame': 1930,
                                 'ext': 'jpg',
                                 'name': 'dove_mul',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'dove_mul/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'dove_mul222/groundtruth_rect.txt',
                                'endFrame': 1297,
                                'ext': 'jpg',
                                'name': 'dove_mul222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'dove_mul222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'elephant222/groundtruth_rect.txt',
                                'endFrame': 2290,
                                'ext': 'jpg',
                                'name': 'elephant222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'elephant222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'fighter_mul/groundtruth_rect.txt',
                                 'endFrame': 2000,
                                 'ext': 'jpg',
                                 'name': 'fighter_mul',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'fighter_mul/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'giraffe_low/groundtruth_rect.txt',
                                 'endFrame': 2267,
                                 'ext': 'jpg',
                                 'name': 'giraffe_low',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'giraffe_low/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'giraffe_motion/groundtruth_rect.txt',
                                 'endFrame': 1500,
                                 'ext': 'jpg',
                                 'name': 'giraffe_motion',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'giraffe_motion/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'giraffe222/groundtruth_rect.txt',
                                'endFrame': 2390,
                                'ext': 'jpg',
                                'name': 'giraffe222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'giraffe222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'ship/groundtruth_rect.txt',
                                 'endFrame': 967,
                                 'ext': 'jpg',
                                 'name': 'ship',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'ship/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'ship_motion/groundtruth_rect.txt',
                                'endFrame': 2301,
                                'ext': 'jpg',
                                'name': 'ship_motion',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'ship_motion/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'star/groundtruth_rect.txt',
                                 'endFrame': 1156,
                                 'ext': 'jpg',
                                 'name': 'star',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'star/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'star_motion/groundtruth_rect.txt',
                                'endFrame': 2122,
                                'ext': 'jpg',
                                'name': 'star_motion',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'star_motion/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'star_mul/groundtruth_rect.txt',
                                 'endFrame': 2160,
                                 'ext': 'jpg',
                                 'name': 'star_mul',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'star_mul/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'star_mul222/groundtruth_rect.txt',
                                'endFrame': 2036,
                                'ext': 'jpg',
                                'name': 'star_mul222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'star_mul222/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'tank_low/groundtruth_rect.txt',
                                'endFrame': 2276,
                                'ext': 'jpg',
                                'name': 'tank_low',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'tank_low/accumulate_events',
                                'startFrame': 1},

                                {'anno_path': 'tower/groundtruth_rect.txt',
                                 'endFrame': 1144,
                                 'ext': 'jpg',
                                 'name': 'tower',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'tower/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'tower333/groundtruth_rect.txt',
                                 'endFrame': 2401,
                                 'ext': 'jpg',
                                 'name': 'tower333',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'tower333/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'truck/groundtruth_rect.txt',
                                 'endFrame': 1131,
                                 'ext': 'jpg',
                                 'name': 'truck',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'truck/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'truck_hdr/groundtruth_rect.txt',
                                 'endFrame': 1969,
                                 'ext': 'jpg',
                                 'name': 'truck_hdr',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': 'truck_hdr/accumulate_events',
                                 'startFrame': 1},

                                {'anno_path': 'whale_mul222/groundtruth_rect.txt',
                                'endFrame': 2171,
                                'ext': 'jpg',
                                'name': 'whale_mul222',
                                'nz': 4,
                                'object_class': 'object',
                                'path': 'whale_mul222/accumulate_events',
                                'startFrame': 1},
                                           ]
        return sequence_info_list


