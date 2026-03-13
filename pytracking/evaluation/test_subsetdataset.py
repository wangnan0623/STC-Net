import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class test_subset(BaseDataset):
    """ test VisEvent dataset


    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.test_subset_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
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
        sequence_info_list = [  {'anno_path': '00335_UAV_outdoor5/groundtruth.txt',
                                'endFrame': 87,
                                'ext': 'jpg',
                                'name': '00335_UAV_outdoor5',
                                'nz': 4,
                                'object_class': 'object',
                                'path': '00335_UAV_outdoor5/event_imgs',
                                'startFrame': 9},

                                {'anno_path': '00340_UAV_outdoor6/groundtruth.txt',
                                 'endFrame': 88,
                                 'ext': 'jpg',
                                 'name': '00340_UAV_outdoor6',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': '00340_UAV_outdoor6/event_imgs',
                                 'startFrame': 1},

                                {'anno_path': '00351_UAV_outdoor6/groundtruth.txt',
                                 'endFrame': 88,
                                 'ext': 'jpg',
                                 'name': '00351_UAV_outdoor6',
                                 'nz': 4,
                                 'object_class': 'object',
                                 'path': '00351_UAV_outdoor6/event_imgs',
                                 'startFrame': 1},


                                           ]
        return sequence_info_list


