# -*- coding: utf-8 -*-
# @Time : 2024/9/3 14:19
# @Author : JohnnyYuan
# @File : face_extract.py
import cv2
import numpy as np
from PIL import Image
import datetime, string ,os, random

def extract_face(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 加载人脸检测分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected!")
        return

    # 选择最大的人脸
    max_face = max(faces, key=lambda x: x[2] * x[3])

    # 提取人脸区域
    face = image[max_face[1]:max_face[1]+max_face[2], max_face[0]:max_face[0]+max_face[3]]

    # 计算正方形的边长
    square_size = max(max_face[2], max_face[3])
    padding = (square_size - max_face[3]) // 2
    padding_y = (square_size - max_face[2]) // 2

    # 创建正方形图片
    square_face = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    square_face[padding_y:padding_y+max_face[2], padding:padding+max_face[3]] = face

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S%f')
    # 随机生成一个固定长度的字符串，例如5位
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    # 拼接图片名称：时间+随机字符+原图片后缀名
    image_filename = f"fake_images/Test-Dev_part_1/Maxhead/DFD{timestamp}_{random_str}{os.path.splitext(image_path)[-1]}"

    # 保存正方形人脸图片
    cv2.imwrite(image_filename, square_face)

    print(f"Face extracted and saved to {image_filename}")

    return image_filename