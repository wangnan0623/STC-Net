import os
import sys
from os.path import join

import cv2
import numpy as np
from dv import AedatFile
from tqdm import tqdm

np.set_printoptions(suppress=True)

# def get_start_frame(seq_name):
#     return pair[seq_name]


def accumulate_events(index, root, seq_name):
    # index: 处理序列的序号，用来展示进度
    # root: /home/exp/data/VisEvent_dataset/VisEvent_test/test_subset/00141_tank_outdoor2
    # seq_name: 00141_tank_outdoor2
    event_data = os.path.join(root, seq_name + '.aedat4')  # /home/exp/data/VisEvent_dataset/VisEvent_test/test_subset/00141_tank_outdoor2/00141_tank_outdoor2.aedat4
    accumulate_path = os.path.join(root, 'accumulate_events_4')  # /home/exp/data/VisEvent_dataset/VisEvent_test/test_subset/00141_tank_outdoor2/accumulate_events

    if not os.path.exists(accumulate_path):
        os.makedirs(accumulate_path)

    with AedatFile(event_data) as f:
        # 读取.aedat文件内事件的相关数据
        pic_shape = f['events'].size  # (260,346)
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        # 将其保存为numpy格式
        event = np.vstack((timestamps, x, y, polarities))
        event = np.swapaxes(event, 0, 1)

        # 获取RGB图像对应的时间戳
        time_series = []
        # get the timestamp of all RGB frames between start_frame and the last frame
        # save the timestamps in time_series
        for frame in f['frames']:
            time_series.append([frame.timestamp_start_of_frame, frame.timestamp_end_of_frame])
        frame_numbers = len(time_series)
        # idx = np.round(np.linspace(0, len(time_series) - 1, int(frame_numbers))).astype(int)
        # time_series = np.array(time_series)[idx]

        # 只保留起始帧和最后一帧之间的数据
        event = event[event[:, 0] >= time_series[0][0]]  # 第一帧的起始时间
        event = event[event[:, 0] < time_series[-1][1]]  # 最后一帧的结束时间

        deal_event_mul(index, event, time_series, pic_shape, accumulate_path)


def deal_event_mul(index, events, frame_intervals, pic_shape, save_name):  # 每帧生成4个子帧
    # 计算总进度（RGB帧数量 × 子帧数量）
    total_frames = (len(frame_intervals) - 1) * 4
    processed_frames = 0

    # 创建进度条
    with tqdm(total=total_frames, desc="{} Processing {} events".format(index, save_name.split('/')[-2])) as pbar:
        for frame_idx in range(len(frame_intervals) - 1):  # 处理每个RGB帧间隔
            start_time, end_time = frame_intervals[frame_idx]

            # 将时间区间分成4个子间隔
            sub_times = np.linspace(start_time, end_time, 5)  # 5个端点，4个区间

            # 为每个子间隔处理事件
            for sub_idx in range(4):
                sub_start = sub_times[sub_idx]
                sub_end = sub_times[sub_idx + 1]

                # 处理当前子间隔的事件
                event_img = np.full(pic_shape, 255, dtype=np.uint8)
                events_in_interval = events[(events[:, 0] >= sub_start) & (events[:, 0] < sub_end)]

                for event in events_in_interval:
                    process_event(event_img, event, pic_shape)

                # 保存事件帧
                filename = f"{frame_idx:04d}_{sub_idx + 1}.jpg"
                cv2.imwrite(os.path.join(save_name, filename), event_img)

                # 更新进度条
                processed_frames += 1
                pbar.update(1)
                pbar.set_postfix({
                    'frame': f'{frame_idx + 1:04d}_{sub_idx + 1}',
                    'events': len(events_in_interval)
                })

def deal_event(index, events, frame_timestamp, pic_shape, save_name):
    i = 1
    # define an empty picture
    # original_img = np.full(pic_shape, 255, dtype=np.uint8)
    original_img = np.full(pic_shape, 255, dtype=np.uint8)

    sub_index = 1
    # accumulate all events between two RGB frames to one event frame
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], 2)

    for event in tqdm(events, desc="{} Writing {} events".format(index, save_name.split('/')[-2])):
        if event[0] >= frame_timestamp[i]:
            cv2.imwrite(save_name + '/' + str(i).zfill(4) + '.jpg', original_img)
            i = i + 1
            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], 2)
            original_img = np.full(pic_shape, 255, dtype=np.uint8)
            sub_index = 1
        elif event[0] < frame_timestamp[i]:
            if event[0] >= sub_frame[sub_index]:
                cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', original_img)
                original_img = np.full(pic_shape, 255, dtype=np.uint8)
                sub_index += 1
            # accumulate events
            process_event(original_img, event, pic_shape)
    # save the event frame
    cv2.imwrite(save_name + '/' + str(i).zfill(4) + '.jpg', original_img)


def process_event(original_img, event, pic_shape):
    # accumulate events at four level
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 < x < pic_shape[1] and 0 < y < pic_shape[0]:
        original_img[y][x] = original_img[y][x] + 1


######################################  main ##############################################
file_name_list = []
frame_numbers = []

# get train sequence name and frame number
file_path = '/home/Data/VisEvent/train_subset'
for file in os.listdir(file_path):
    file_name_list.append(file)
file_name_list.sort()

for index, seq_name in enumerate(file_name_list):
    file_path = '/home/Data/VisEvent/train_subset'
    data_path = os.path.join(file_path, seq_name)
    accumulate_path = os.path.join(data_path, 'accumulate_events_4')

    # # 若文件已存在，则删除文件
    # if os.path.exists(accumulate_path):
    #     try:
    #         for filename in os.listdir(accumulate_path):
    #             file_path = os.path.join(accumulate_path, filename)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)
    #             elif os.path.isdir(file_path):
    #                 os.rmdir(file_path)
    #         print(f"'{accumulate_path}'中的所有文件已被删除")
    #     except Exception as e:
    #         print(f"清空'{accumulate_path}'时发生错误：{e}")

    # 如果目标文件已经存在，则跳过该序列
    if os.path.exists(accumulate_path):
        if 4 * len(os.listdir(join(data_path, 'vis_imgs_jpg'))) == len(os.listdir(accumulate_path)):
            print(f"Skipping {seq_name} as {accumulate_path} already exists.")
            continue

    accumulate_events(index, data_path, seq_name)

# # get test sequence name and frame number
# file_path = '/home/Data2/VisEvent/test_subset'
# for file in os.listdir(file_path):
#     file_name_list.append(file)
# file_name_list.sort()
#
# for index, seq_name in enumerate(file_name_list):
#     file_path = '/home/Data2/VisEvent/test_subset'
#     data_path = os.path.join(file_path, seq_name)
#     accumulate_path = os.path.join(data_path, 'accumulate_events')
#
#     # 若文件已存在，则删除文件
#     if os.path.exists(accumulate_path):
#         try:
#             for filename in os.listdir(accumulate_path):
#                 file_path = os.path.join(accumulate_path, filename)
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#                 elif os.path.isdir(file_path):
#                     os.rmdir(file_path)
#             print(f"'{accumulate_path}'中的所有文件已被删除")
#         except Exception as e:
#             print(f"清空'{accumulate_path}'时发生错误：{e}")
#
#     accumulate_events(index, data_path, seq_name)