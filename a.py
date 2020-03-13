
import numpy as np
# 导包
import os
import cv2
import sys
from PIL import Image
def getImageAndLabels(path):
    """
    获取图片特征值和目标值
    :param path:
    :return:
    """
    """
    PIL(Python Image Library)是python的第三方图像处理库，但是由于其强大的功能与众多的使用人数，
    几乎已经被认为是python官方图像处理库了。
    Image模块是在Python PIL图像处理中常见的模块，对图像进行基础操作的功能基本都包含于此模块内。
    如open、save、conver、show…等功能
    """
    facesSamples = []
    ids = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier("D:/software/anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    #遍历列表中的图片
    for imagePath in imagePaths:
        #打开图片
        PIL_img = Image.open(imagePath).convert("L")
        #将图像转换我数组
        img_numpy = np.array(PIL_img,"uint8")
        print(img_numpy.shape)
        faces = face_detector.detectMultiScale(img_numpy)
        #获取每张图片的id
        id = int(os.path.split(imagePath)[1].split(".")[0])
        for x,y,w,h in faces:
            #添加人脸区域图片
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    print(ids)
    print(facesSamples)
    return facesSamples,ids
#图片路径
path = "../data/jm"
#获取图像数组和id数组标签
faces,ids = getImageAndLabels(path)
#训练对象
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces,np.array(ids))
#保存训练文件
# recognizer.write("./trainer.yml")