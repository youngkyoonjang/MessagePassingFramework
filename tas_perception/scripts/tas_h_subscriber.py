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


class TAS_Integrator_YJ:
    def __init__(self):
        # params

        self.img = None
        self.img_tmp = None
        self.instance_json = None
        self.body_json = None
        self.hand_json = None
        
        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"

        # json_topic_instance = "/instance_json_out"
        # rospy.Subscriber(json_topic_instance, String, self.instance_json_callback)
        
        # json_topic_body = "/body_json_out"
        # rospy.Subscriber(json_topic_body, String, self.body_json_callback)
        
        json_topic_hand = "/hand_state_json_out"
        rospy.Subscriber(json_topic_hand, String, self.hand_json_callback)
        
        ########################################################################## hand initiation
        self.color_rgb = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
        self.color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


        self.hand_rgb = [(0, 90, 181), (220, 50, 32)] 
        self.hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]

        self.obj_rgb = (255, 194, 10)
        self.obj_rgba = (255, 194, 10, 70)


        self.side_map = {'l':'Left', 'r':'Right'}
        self.side_map2 = {0:'Left', 1:'Right'}
        self.side_map3 = {0:'L', 1:'R'}
        self.state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
        self.state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}

        # font
        self.font = cv2.FONT_HERSHEY_COMPLEX
        # fontScale
        self.fontScale = 0.7
        # Blue color in BGR
        self.color = (0, 0, 0)
        # Line thickness of 2 px
        self.thickness = 2
        ##########################################################################
                
        
        color_sub = message_filters.Subscriber(image_topic, Image)
        depth_sub = message_filters.Subscriber(aligned_depth_to_color, Image)
        message_filters.Subscriber(json_topic_hand, String)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)
        
        self.pub = rospy.Publisher('memory_json_out', String, queue_size=10)

        ts.registerCallback(self.image_callback)

    def instance_json_callback(self, msg):
        try:
            loaded_dictionary_ins = json.loads(msg.data)
        except CvBridgeError:
            print("e")
        else:
            self.instance_json = loaded_dictionary_ins

    def body_json_callback(self, msg):
        try:
            loaded_dictionary_body = json.loads(msg.data)
        except CvBridgeError:
            print("e")
        else:
            self.body_json = loaded_dictionary_body
            
    def hand_json_callback(self, msg):
        try:
            loaded_dictionary_hand = json.loads(msg.data)

            self.hand_json = loaded_dictionary_hand
            
            integrated_data_tmp_for_saving={}
        except CvBridgeError:
            print("e")
        else:
            
            if(self.hand_json!=None):
                self.img = self.img_tmp
                ## yjang inserted for visualising Hand-Object State Estimation output from the integrated results

              

                # self.hand_states = integrated_data[i_cnt]['hand_states']
                mask_tmp_ = []
                # if(0<len(self.hand_states['hands'])):
                if(0<len(self.hand_json[0]['hands'])):  
                    for i in range(0, len(self.hand_json[0]['hands'])):
                        individual_hand_info_tmp = self.hand_json[0]['hands'][i]
                        hand_bbox = individual_hand_info_tmp['bbox']
                        hand_state = individual_hand_info_tmp['state']
                        hand_side = individual_hand_info_tmp['side_id']

                        cv2.rectangle(self.img, (hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3]), color=self.hand_rgb[hand_side], thickness=4)
                        cv2.rectangle(self.img, (hand_bbox[0], max(0, hand_bbox[1]-30)), (hand_bbox[0]+62, max(0, hand_bbox[1]-30)+30), color=self.hand_rgb[hand_side], thickness=4)

                        text = f'{self.side_map3[int(float(hand_side))]}-{self.state_map2[int(float(hand_state))]}'
                        # org
                        org = (hand_bbox[0]+6, max(0, hand_bbox[1])-8)
                        # Using cv2.putText() method
                        cv2.putText(self.img, text, org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

                    for i in range(0, len(self.hand_json[0]['objects'])):
                        individual_object_info_tmp = self.hand_json[0]['objects'][i]
                        object_bbox = individual_object_info_tmp['bbox']

                        cv2.rectangle(self.img, (object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3]), color=self.obj_rgb, thickness=4)
                        cv2.rectangle(self.img, (object_bbox[0], max(0, object_bbox[1]-30)), (object_bbox[0]+32, max(0, object_bbox[1]-30)+30), color=self.obj_rgb, thickness=4)

                        text = f'O'
                        # org
                        org = (object_bbox[0]+5, max(0, object_bbox[1])-8)
                        # Using cv2.putText() method
                        cv2.putText(self.img, text, org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                    #########################################
                    
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
            print("error in hands")
        else:
            self.img_tmp = cv2_img

                
    def start(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                cv2.namedWindow('integrator_hand')        # Create a named window
                cv2.moveWindow('integrator_hand', 1500,600)  # Move it to (40,30)

                cv2.imshow('integrator_hand', self.img)

                key = cv2.waitKey(1)

           
            self.loop_rate.sleep()


def main():
    rospy.init_node('tas_integrator_h', anonymous=True)

    Ctas_Integrator_YJ = TAS_Integrator_YJ()

    Ctas_Integrator_YJ.start()


if __name__ == '__main__':
    main()


