# -*- coding: utf-8 -*-
# @Time : 2024/7/5 14:31
# @Author : JohnnyYuan
# @File : run.py
from transformers import pipeline
import time


model_str = "DhruvJariwala_image"

pipe = pipeline('image-classification', model=model_str, device=0)
start_time = time.time()
print(pipe('fake_images/__results___28_0.png'))  # 放一张图片的路径，任何尺寸都行
end_time = time.time()

print('time:',end_time-start_time)
