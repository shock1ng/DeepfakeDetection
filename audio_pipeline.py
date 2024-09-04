# -*- coding: utf-8 -*-
# @Time : 2024/7/19 13:48
# @Author : JohnnyYuan
# @File : audio_pipeline.py

from transformers import pipeline, Wav2Vec2ForCTC
import time

pipe = pipeline("audio-classification", model="deepfake_audio_heem2")

result = pipe('audio2test/luvvoice.com-20240719-eBm1.mp3')

print(result)
max_score = 0
max_label = ''

# 遍历列表，找到最高分的label
for item in result:
    if item['score'] > max_score:
        max_score = item['score']
        max_label = item['label']
if max_label == 'AIVoice':
    max_label = 'fake'
else:
    max_label = 'genuine'

print(f"The label with the highest score is: {max_label}")