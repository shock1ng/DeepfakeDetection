# -*- coding: utf-8 -*-
# @Time : 2024/9/2 15:18
# @Author : JohnnyYuan
# @File : imageModel_test.py
import os
from transformers import pipeline
import json
from face_extract import extract_face  # 返回头像路径

# 指定要读取的文件名
output_file = '/home/hd/JohnnyYuan/DFD/Test-Dev-{file_name-cate}.json'

# 读取 JSON 文件并转换为字典
with open(output_file, 'r') as file:
    file_to_category = json.load(file)

# 把图片文件名都读取到文件夹中，设置你的文件夹路径
folder_path = 'fake_images/Test-Dev_part_1/Test-Dev'
# 初始化一个空列表来存储文件名
image_files = []
# 使用os.listdir()列出文件夹中的所有文件和文件夹
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.jpg
    if filename.endswith('.jpg'):
        # 如果是，将文件名添加到列表中
        image_files.append(filename)

pipe = pipeline("image-classification", model="DhruvJariwala_image", device=0)  # model这里是你本地下载的Hugging face模型的路径
# 打印文件列表
true = 0
now = 0
for item in image_files:
    now += 1
    face_path = extract_face(f'fake_images/Test-Dev_part_1/Test-Dev/{item}')   # 提取面部
    result = pipe(f"/home/hd/JohnnyYuan/DFD/{face_path}")   # 获得结果
    max_score = 0
    max_label = 0
    # 遍历列表，找到最高分的label
    for item2 in result:
        if item2['score'] > max_score:
            max_score = item2['score']
            max_label = item2['label']
    if max_label == 'Fake':
        max_label = 1
    else:
        max_label = 0

    if file_to_category[item] == max_label:
        true += 1
        print(f'Now ACC = {true/now}  {now}/{len(image_files)}')

print('最后测得的准确度是',true/len(image_files))

