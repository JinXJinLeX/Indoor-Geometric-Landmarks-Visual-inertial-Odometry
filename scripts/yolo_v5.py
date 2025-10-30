#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
ros_path='/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import torch

import numpy as np
import random

import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import rospy

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes

# 20240330_xjl
# import rosbag
# from cv_bridge import CvBridge
# from cv_bridge import CvBridgeError
# image_path='/media/scott/Studio/z/240307data/extract/' #要存放图片的位置

# video_writer=None
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_save_path = os.path.expanduser("~/output_video.mp4")
# video_writer = cv2.VideoWriter(video_save_path,fourcc,30.0,(1280,720))

class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=10)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image',  Image, queue_size=5)

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        # 20230424_xjl
        # self.boundingBoxes.encoding = image.dtype.name
        self.boundingBoxes.width=image.width
        self.boundingBoxes.height=image.height
        # self.boundingBoxes.encoding=image.dtype.name
        self.boundingBoxes.encoding="rgb8"
        self.boundingBoxes.is_bigendian=image.is_bigendian
        self.boundingBoxes.step = image.step
        self.boundingBoxes.data=image.data

        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)

        # t0 = rospy.Time.now().to_nsec()
        results = self.model(self.color_image)

        # video_writer.write(self.color_image)

        # t1 = rospy.Time.now().to_nsec()
        # rospy.loginfo(t1 - t0)
        
        # xmin    ymin    xmax   ymax  confidence  class    name
    
        boxs = results.pandas().xyxy[0].values
        self.dectshow(self.color_image, boxs, image.height, image.width)
        # cv2.waitKey(3)


    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()

        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = box[-1]

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(img, box[-1],
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


            self.boundingBoxes.bounding_boxes.append(boundingBox)
        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(img, height, width)
        # img = cv2.resize(img,None,fx=0.75,fy=0.75)
        # cv2.imshow('YOLOv5', img)

        # 定义保存图像的目录
        # output_dir = "/home/scott/gvins_yolo_output/yolo_result"
        # 保存图像到文件
        # filename = os.path.join(output_dir, f"{random.random()}.jpg")
        # cv2.imwrite(filename, img)
        # 递增计数器
        
    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


# class ImageCreator():
#     def __init__(self):
#         self.bridge = CvBridge()
#         with rosbag.Bag('/media/scott/Studio/z/240307data/240307_1_2024-03-07-16-49-50.bag', 'r') as bag:   #要读取的bag文件；
#             for topic,msg,t in bag.read_messages():
#                 if topic == "/camera/color/image_raw":  #图像的topic；
#                         try:
#                             cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
#                         except CvBridgeError as e:
#                             print(e)
#                         timestr = "%.9f" %  msg.header.stamp.to_sec()
#                         #%.9f表示小数点后带有9位，可根据精确度需要修改；
#                         image_name = timestr+ ".png" #图像命名：时间戳.jpg
#                         cv2.imwrite(image_path+image_name, cv_image)  #保存；

def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    # t0 = rospy.Time.now().to_nsec()
    yolo_dect = Yolo_Dect()
    # t1 = rospy.Time.now().to_nsec()
    # rospy.loginfo(t1 - t0)
    # print(t1 - t0)
    rospy.spin()


if __name__ == "__main__":
    main()
# 20240330_xjl
    # try:
    #     image_creator = ImageCreator()
    # except rospy.ROSInterruptException:
    #     pass
