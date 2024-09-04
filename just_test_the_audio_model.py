# -*- coding: utf-8 -*-
# @Time : 2024/8/16 15:12
# @Author : JohnnyYuan
# @File : just_test_the_audio_model.py
import os
from transformers import pipeline
from tqdm import tqdm
# 假设你的文件名为 'data.txt'
filename = 'audio2test/track1_label.txt'

# 初始化一个空字典来存储结果
file_dict = {}

# 打开文件并读取内容
with open(filename, 'r') as file:
    for line in file:
        # 去除每行的首尾空白字符，然后分割行以获取文件名和标签
        parts = line.strip().split()
        if len(parts) == 2:
            file_name, label = parts
            # 将文件名和标签添加到字典中
            file_dict[file_name] = label


# 把语音文件名都读取到文件夹中，设置你的文件夹路径
folder_path = 'audio2test/sceneFake_not4DFD'
# 初始化一个空列表来存储文件名
wav_files = []
# 使用os.listdir()列出文件夹中的所有文件和文件夹
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.wav
    if filename.endswith('.wav'):
        # 如果是，将文件名添加到列表中
        wav_files.append(filename)

pipe = pipeline("audio-classification", model="davidcombei-base-utcn")
# 打印文件列表
true = 0
now = 0
for item in wav_files:
    now += 1
    result = pipe(f'audio2test/track1test/{item}')
    max_score = 0
    max_label = ''
    # 遍历列表，找到最高分的label
    for item2 in result:
        if item2['score'] > max_score:
            max_score = item2['score']
            max_label = item2['label']
    if max_label == 'AIVoice':
        max_label = 'fake'
    else:
        max_label = 'genuine'
    if file_dict.get(str(item)) == max_label:
        true += 1
        print(f'Now ACC = {true/now}  {now}/{len(wav_files)}')

print('最后测得的准确度是',true/len(wav_files))

