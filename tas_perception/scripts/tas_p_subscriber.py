import os
from argparse import ArgumentParser

import cv2
from PIL import Image


# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import String
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError


import json
from datetime import timedelta
import time

import numpy as np


import os

import message_filters
from sensor_msgs.msg import Image, CameraInfo

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


class TAS_Object_Subscriber_YJ:
    def __init__(self):
        # params

        self.img = None
        self.img_tmp = None
        self.body_json = None
        
        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"

        json_topic_body = "/body_json_out"
        rospy.Subscriber(json_topic_body, String, self.body_json_callback)
        
        
        ########################################################################## instance initiation
        # self.det_cat_id = args.det_cat_id
        self.bbox_thr = 0.3
        self.kpt_thr = 0.3
        self.radius = 4
        self.thickness_ = 1
        
        # font
        self.font = cv2.FONT_HERSHEY_COMPLEX
        # fontScale
        self.fontScale = 0.7
        # Blue color in BGR
        self.color = (0, 0, 255)
        # Line thickness of 2 px
        self.thickness = 2
        ##########################################################################
                
        
        color_sub = message_filters.Subscriber(image_topic, Image)
        depth_sub = message_filters.Subscriber(aligned_depth_to_color, Image)
        message_filters.Subscriber(json_topic_body, String)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)
        
        

        ts.registerCallback(self.image_callback)

    def body_json_callback(self, msg):
        try:
            loaded_dictionary_body = json.loads(msg.data)
            self.body_json = loaded_dictionary_body
        except CvBridgeError:
            print("e")
        else:
            
            if(self.body_json!=None):
                self.img = self.img_tmp
                 
                mask_tmp_ = []
                # if(0<len(self.hand_states['hands'])):
                if(0<len(self.body_json[0]['body'])):            
                    integrated_data_tmp_for_saving=self.body_json[0]
                    integrated_data = [integrated_data_tmp_for_saving]
                    # body = integrated_data[0]['body']
                    body_pose_results = integrated_data[0]['body']
                 
                    #########################################
                    for i in range(0, len(body_pose_results)):
                        bbox = body_pose_results[i]['bbox']
                        keypoints = body_pose_results[i]['keypoints']
                                                
                        cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=1)
                        
                        #face
                        for j in range(1, 3):
                            cv2.line(self.img, (int(keypoints[j-1][0]), int(keypoints[j-1][1])), (int(keypoints[j][0]), int(keypoints[j][1])), (0,255,0), 1)
                        cv2.line(self.img, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[2][0]), int(keypoints[2][1])), (0,255,0), 1)
                        
                        ##two arms
                        for j in range(2, 10, 2): 
                            #right
                            str_idx = j
                            end_idx = j+2
                            if(0.5<=keypoints[end_idx][2]):
                                cv2.line(self.img, (int(keypoints[str_idx][0]), int(keypoints[str_idx][1])), (int(keypoints[end_idx][0]), int(keypoints[end_idx][1])), (0,255,0), 1)
                            
                            #left
                            str_idx = j-1
                            end_idx = (j+2)-1
                            if(0.5<=keypoints[end_idx][2]):
                                cv2.line(self.img, (int(keypoints[str_idx][0]), int(keypoints[str_idx][1])), (int(keypoints[end_idx][0]), int(keypoints[end_idx][1])), (0,255,0), 1)
                        
                        ##two legs
                        for j in range(12, 16, 2):
                            #right
                            str_idx = j
                            end_idx = j+2
                            if(0.5<=keypoints[end_idx][2]):
                                cv2.line(self.img, (int(keypoints[str_idx][0]), int(keypoints[str_idx][1])), (int(keypoints[end_idx][0]), int(keypoints[end_idx][1])), (0,255,0), 1)
                            
                            #left
                            str_idx = j-1
                            end_idx = (j+2)-1
                            if(0.5<=keypoints[end_idx][2]):
                                cv2.line(self.img, (int(keypoints[str_idx][0]), int(keypoints[str_idx][1])), (int(keypoints[end_idx][0]), int(keypoints[end_idx][1])), (0,255,0), 1)
                        
                        ##torso
                        cv2.line(self.img, (int(keypoints[5][0]), int(keypoints[5][1])), (int(keypoints[6][0]), int(keypoints[6][1])), (0,255,0), 1)
                        cv2.line(self.img, (int(keypoints[6][0]), int(keypoints[6][1])), (int(keypoints[12][0]), int(keypoints[12][1])), (0,255,0), 1)
                        cv2.line(self.img, (int(keypoints[12][0]), int(keypoints[12][1])), (int(keypoints[11][0]), int(keypoints[11][1])), (0,255,0), 1)
                        cv2.line(self.img, (int(keypoints[11][0]), int(keypoints[11][1])), (int(keypoints[5][0]), int(keypoints[5][1])), (0,255,0), 1)
                        
                        ## all joints
                        for j in range(0, len(keypoints)):
                            if(0.5<=keypoints[j][2]):
                                cv2.circle(self.img, (int(keypoints[j][0]), int(keypoints[j][1])), 3, (0,255,0), -1)
                        
    # from dateutil.parser import parse
    # def image_callback(msg1, msg2, msg3):
    def image_callback(self, msg1, msg2):
        try:
            # Convert your ROS Image message to OpenCV2 for color
            color_img = self.bridge.imgmsg_to_cv2(msg1, "bgr8")
            cv2_img = cv2.resize(color_img, (int(color_img.shape[1]/2), int(color_img.shape[0]/2)))

            
            # Convert your ROS Image message to OpenCV2 for depth
            depth_image = self.bridge.imgmsg_to_cv2(msg2, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)

            # grey_color = 153
            depth_image_3d = np.dstack((depth_array,depth_array,depth_array)) #depth image is 1 channel, color is 3 channels
            cv2_depth_img = cv2.resize(depth_image_3d, (int(depth_image_3d.shape[1]/2), int(depth_image_3d.shape[0]/2)))
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)

            depth_img = depth_colormap


        except CvBridgeError:
            print("error in pose")
        else:
            self.img_tmp = cv2_img
            
                
    def start(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                cv2.namedWindow('integrator_pose')        # Create a named window
                cv2.moveWindow('integrator_pose', 1500,600)  # Move it to (40,30)

                cv2.imshow('integrator_pose', self.img)
                
                
                key = cv2.waitKey(1)
                
            self.loop_rate.sleep()


def main():
    rospy.init_node('tas_integrator_p', anonymous=True)

    Ctas_object_subscriber_YJ = TAS_Object_Subscriber_YJ()

    Ctas_object_subscriber_YJ.start()


if __name__ == '__main__':
    main()


