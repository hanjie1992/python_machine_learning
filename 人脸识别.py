import cv2


def show_image():
    """
    OpenCV读取图像
    :return:
    """
    # 导包
    import cv2
    # 读取照片
    img = cv2.imread("../data/lena.jpg")  # 路径中不能有中文
    # 显示图片
    cv2.imshow("read_img", img)
    # 输入毫秒值，传0就是无限等待
    cv2.waitKey(3000)
    # 释放内存,由于OpenCV底层是C++写的
    cv2.destroyAllWindows()
    return None


def gray_level():
    """
    OpenCV进行灰度转换
    :return:
    """
    import cv2
    img = cv2.imread("../data/lena.jpg")
    cv2.imshow("BGR_IMG", img)
    # 将图片转化为灰度
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 展示图片
    cv2.imshow("gray_img", gray_img)
    # 设置展示时间
    cv2.waitKey(3000)
    # 保存图片
    cv2.imwrite("./gray_lena.jpg", gray_img)
    # 释放内存
    cv2.destroyAllWindows()
    return None


def size_image():
    import cv2
    # 读取图例
    img = cv2.imread("../data/lena.jpg")
    # 展示图像
    cv2.imshow("img", img)
    print("原来图片的形状", img.shape)
    # 设置图片的形状
    resize_img = cv2.resize(img, dsize=(600, 600))
    print("修改后的形状", resize_img.shape)
    cv2.imshow("resize_img", resize_img)
    # 设置等待时间
    cv2.waitKey(0)
    # 释放内存
    cv2.destroyAllWindows()
    return None


def draw_image():
    """
    OpenCV画图，对图片进行编辑
    :return:
    """
    # 导包
    import cv2
    img = cv2.imread("../data/lena.jpg")
    # 左上角的坐标是(x,y),矩形的宽度和高度是(w,h)
    x, y, w, h = 100, 100, 100, 100
    # 绘制矩形
    cv2.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=2)
    # 绘制圆
    x, y, r = 200, 200, 100
    cv2.circle(img, center=(x, y), radius=r, color=(0, 0, 255), thickness=2)
    # 显示图片
    cv2.imshow("rectangle_img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def out_face_detect_static():
    def face_detect_static():
        """
        静态人脸检测
        :return:
        """
        # 导包
        import cv2
        # 图片转化为灰度图片
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 加载特征数据
        face_detector = cv2.CascadeClassifier(
            "C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray_image)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.imshow("result", img)

    # 加载图片
    img = cv2.imread('../data/lena.jpg')
    # 调用人脸检测方法
    face_detect_static()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def out_many_face_detect_static():
    import cv2
    def many_face_detect_static():
        # 图片进行灰度处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 加载数据特征
        face_detector = cv2.CascadeClassifier(
            "C:/ProgramData/Anaconda3/Lib/site-packages/cv2"
            "/data/haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray)
        for x, y, w, h in faces:
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          color=(0, 0, 255), thickness=2)
            cv2.circle(img, (x + w // 2, y + w // 2),
                       radius=w // 2, color=(0, 255, 0), thickness=2)
        cv2.imshow("result", img)

    # 加载图片
    img = cv2.imread("../data/face3.jpg")
    # 调用人脸检测方法
    many_face_detect_static()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def video_face_detect():
    """
    视频人脸检测
    :return:
    """

    def face_detect(frame):
        # 将图片进行灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 加载特征数据
        face_detector = cv2.CascadeClassifier(
            "C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.circle(frame, center=(x + w // 2, y + h // 2), radius=(w // 2), color=(0, 255, 0), thickness=2)
        cv2.imshow("result", frame)

    video_face = cv2.VideoCapture("G:/video.mp4")
    while True:
        # read()方法返回视频中检测的对象，视频在播放flag为True,frame为当前帧上的图片
        flag, frame = video_face.read()
        print("flag:", flag, "frame.shape:", frame.shape)
        if not flag:
            break
        face_detect(frame)
        cv2.waitKey(20)
    cv2.destroyAllWindows()
    video_face.release()


def out_getImageAndLabels():
    import numpy as np
    def getImageAndLabels(path):
        """
        获取图片特征值和目标值
        :param path:
        :return:
        """
        # 导包
        import os
        import cv2
        import sys
        from PIL import Image
        """
        PIL(Python Image Library)是python的第三方图像处理库，
        但是由于其强大的功能与众多的使用人数，几乎已经被认为
        是python官方图像处理库了。Image模块是在Python PIL图像
        处理中常见的模块，对图像进行基础操作的功能基本都包含于此模块内。
        如open、save、conver、show…等功能
        """
        facesSamples = []
        ids = []
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        face_detector = cv2.CascadeClassifier(
            "C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data"
            "/haarcascade_frontalface_default.xml")
        # 遍历列表中的图片
        for imagePath in imagePaths:
            # 打开图片
            PIL_img = Image.open(imagePath).convert("L")
            # 将图像转换我数组
            img_numpy = np.array(PIL_img, "uint8")
            faces = face_detector.detectMultiScale(img_numpy)
            # 获取每张图片的id
            id = int(os.path.split(imagePath)[1].split(".")[0])
            for x, y, w, h in faces:
                # 添加人脸区域图片
                facesSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return facesSamples, ids

    # 图片路径
    path = "../data/jm"
    # 获取图像数组和id数组标签
    faces, ids = getImageAndLabels(path)
    # 训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    # 保存训练文件
    recognizer.write("./trainer.yml")


def match_face():
    """
    人脸匹配
    :return:
    """
    # 导包
    import cv2
    import numpy as np
    import os
    # 加载训练数据集文件
    recogizer = cv2.face.LBPHFaceRecognizer_create()
    recogizer.read("./trainer.yml")
    # 准备识别的图片
    img = cv2.imread("../data/8.pgm")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(
        "C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 人脸识别
        id, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        print("标签id：", id, "置信度评分：", confidence)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    人脸识别：基于OpenCV的人脸识别
        OpenCV全程Open Source Computer Vision Library，是一个跨平台的计算机视觉库。用C++语言编写
        安装：pip install opencv-python 
    """
    """
    OpenCV用法1：显示图像
        API：
            imshow() 显示图像
            waitkey() 设置图片显示的时长，单位毫秒，0无限等待，为一直显示
    """
    # show_image()

    """
    OpenCV用法2:图片灰度转换。减少计算量
    """
    # gray_level()

    """
    OpenCV用法3：修改图片尺寸
    """
    # size_image()

    """
    OpenCV用法4：画图
    
    """
    # draw_image()

    """
    OpenCV用法5：静态人脸检测
    摄影作品可能包含很多令人愉悦的细节。 但是， 由于灯光、 视角、 视距、 摄像头抖动以及数字噪声的变化， 图像细节变得不稳定。
    Haar级联数据
    """
    # out_face_detect_static()
    # out_many_face_detect_static()

    """
    OpenCV用法6：视频人脸检测
    """
    # video_face_detect()

    """
    OpenCV用法7：LBPH人脸识别，预测人脸归属
        LBPH（Local Binary Pattern Histogram） 将检测到的人脸分为小单元，
        并将其与模型中的对应单元进行比较， 对每个区域的匹配值产生一个直方图。
    """
    # out_getImageAndLabels()
    # match_face()
