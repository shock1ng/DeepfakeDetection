# -*- coding: utf-8 -*-
# @Time : 2024/9/3 11:14
# @Author : JohnnyYuan
# @File : Json读取.py
import json
import os
# 这个.py文件仅针对 OpenForensics数据集
# OpenForensics数据集标签文件来源 https://zenodo.org/records/5528418

# 设定你的json文件路径
json_file = 'Test-Dev_poly.json'

# 读取 JSON 文件
with open(json_file, 'r') as file:
    data = json.load(file)

# 初始化一个空字典来存储结果
file_to_category = {}

# 遍历 'images' 列表
for image in data['images']:
    file_name = image['file_name']
    image_id = image['id']

    # 从文件名中提取纯粹的文件名部分
    base_name = os.path.basename(file_name)

    # 遍历 'annotations' 列表，找到匹配的 'image_id'
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # 如果找到匹配的 'image_id'，则将提取的文件名和 'category_id' 添加到字典中
            file_to_category[base_name] = annotation['category_id']
            break  # 找到匹配项后，不需要继续遍历

# 指定要保存的文件名
output_file = 'Test-Dev-{file_name-cate}.json'  # 保存后亲自打开看看里面是什么你就明白了

# 将字典保存到 JSON 文件
with open(output_file, 'w') as file:
    json.dump(file_to_category, file, indent=4)  # 使用 indent 参数来美化输出

print(f'字典已保存到 {output_file}')