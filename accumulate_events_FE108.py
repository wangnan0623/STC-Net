import os
import sys
from os.path import join

import cv2
import numpy as np
from dv import AedatFile
from tqdm import tqdm

np.set_printoptions(suppress=True)

def get_start_frame(seq_name):
    return pair[seq_name]


def accumulate_events(index, root, frame_number):
    seq_name = os.path.basename(root)  # E:\\FE108\\train\\airplane
    event_data = os.path.join(root, 'events.aedat4')  # E:\\FE108\\train\\airplane\\events.aedat4
    accumulate_path = os.path.join(root, 'accumulate_events_4')  # E:\\FE108\\train\\airplane\\accumulate_events
    if not os.path.exists(accumulate_path):
        os.makedirs(accumulate_path)
    start_frame = get_start_frame(seq_name)
    with AedatFile(event_data) as f:
        pic_shape = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        event = np.vstack((timestamps, x, y, polarities))
        event = np.swapaxes(event, 0, 1)
        time_series = []
        count = 0
        # get the timestamp of all RGB frames between start_frame and the last frame
        # save the timestamps in time_series
        for frame in f["frames"]:
            count += 1
            if count >= start_frame and count <= start_frame + frame_number:
                time_series.append(frame.timestamp_start_of_frame)
            else:
                continue
        # only keep events happened after strat_frame
        event = event[event[:, 0] >= time_series[0]]
        event = event[event[:, 0] < time_series[-1]]
        # deal events data
        deal_event_mul(index, event, time_series, pic_shape, accumulate_path)

def deal_event_mul(index,events, frame_timestamp, pic_shape, save_name):  # 每帧生成3个子帧
    i = 1
    # 创建一副填充为白色（值为255）的图像original_img，尺寸与事件尺寸pic_shape一致
    original_img = np.full(pic_shape, 255, dtype=np.uint8)
    sub_index = 1
    # 使用np.linspace生成一个包含五个元素的数组，这些元素均匀分布于frame_timestamp的第一个和第二个时间戳之间
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], 5)

    for event in tqdm(events, desc="{} Writing {} events ".format(index, save_name.split('/')[-2])):
        if event[0] >= frame_timestamp[i]:
            cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', original_img)  # 保存最后一个子帧
            i = i + 1
            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], 5)  # 将子帧的时间戳移动到下一个RGB区间
            original_img = np.full(pic_shape, 255, dtype=np.uint8)  # 重置原始图像
            sub_index = 1  # 重置子帧索引
        elif event[0] < frame_timestamp[i]:
            if event[0] >= sub_frame[sub_index]:
                cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', original_img)  # 保存当前子帧
                original_img = np.full(pic_shape, 255, dtype=np.uint8)  # 重置原始图像
                sub_index = sub_index + 1  # 累加子帧索引
            process_event(original_img, event, pic_shape)
    # 循环结束，保存最后一帧的最后一个子帧
    if i < len(frame_timestamp) and sub_index <= 4:
        # 只保存最后一个有事件内容的子帧
        cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', original_img)



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

pair = {}
match_file = '/home/Data/FE240/pair.txt'
with open(match_file, 'r') as f:
    for line in f.readlines():
        file, start_frame = line.split()
        pair[file] = int(start_frame) + 1

file_name_list = []
frame_numbers = []

# get train sequence name and frame number
file = '/home/Data/FE240/train_frame_counts.txt'
with open(file, 'r') as f:
    for line in f:
        parts = line.strip().split(': ')
        if len(parts) == 2:
            file_name = parts[0]
            frame_count = parts[1]
            file_name_list.append(file_name)
            frame_numbers.append(int(frame_count))

for index, i in enumerate(file_name_list):
    data = os.path.join('/home/Data/FE240/train', i)
    frame_number = frame_numbers[index]

    accumulate_path = os.path.join(data, 'accumulate_events_4')

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
        if 4 * len(os.listdir(join(data, 'img'))) == len(os.listdir(accumulate_path)):
            print(f"Skipping {i} as {accumulate_path} already exists.")
            continue

    accumulate_events(index, data, frame_number)

# # get test sequence name and frame number
# file = '/home/Data2/FE240/test_frame_counts.txt'
# with open(file, 'r') as f:
#     for line in f:
#         parts = line.strip().split(': ')
#         if len(parts) == 2:
#             file_name = parts[0]
#             frame_count = parts[1]
#             file_name_list.append(file_name)
#             frame_numbers.append(int(frame_count))
#
# for index, i in enumerate(file_name_list):
#     data = os.path.join('/home/Data2/FE240/test', i)
#     frame_number = frame_numbers[index]
#
#     accumulate_path = os.path.join(data, 'accumulate_events')
#
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
#     accumulate_events(index, data, frame_number)