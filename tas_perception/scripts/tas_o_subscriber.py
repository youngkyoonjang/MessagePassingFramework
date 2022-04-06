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


class TAS_Object_Subscriber_YJ:
    def __init__(self):
        # params

        self.img = None
        self.img_tmp = None
        self.instance_json = None
        
        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"

        json_topic_instance = "/instance_json_out"
        rospy.Subscriber(json_topic_instance, String, self.instance_json_callback)
        
        
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
        message_filters.Subscriber(json_topic_instance, String)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)
        
        

        ts.registerCallback(self.image_callback)

    def instance_json_callback(self, msg):
        try:
            loaded_dictionary_ins = json.loads(msg.data)
            self.instance_json = loaded_dictionary_ins
        except CvBridgeError:
            print("e")
        else:
            if(self.instance_json!=None):
                self.img = self.img_tmp
                 
                
                mask_tmp_ = []
                # if(0<len(self.hand_states['hands'])):
                if(0<len(self.instance_json[0]['instances'])):            
                    integrated_data_tmp_for_saving=self.instance_json[0]
                    integrated_data = [integrated_data_tmp_for_saving]
                    # body = integrated_data[0]['body']
                    instances_results = integrated_data[0]['instances']
                    
                    
                    for i in range(0, len(instances_results['labels'])):
                        label = instances_results['labels'][i]
                        pred_box = instances_results['pred_boxes'][i]
                        score = instances_results['scores'][i]
                        pred_class = instances_results['pred_classes'][i]
                        pred_contour = instances_results['pred_contours'][i]
                        
                        countour_mask_array = [np.array(pred_contour).astype(np.int32)]
                        #create an empty image for contours                    
                        image_binary = np.zeros((self.img.shape), np.uint8)
                        # draw the contours on the empty image
                        cv2.drawContours(self.img, countour_mask_array, -1, (0,255,0), 1)
                        
                        cv2.rectangle(self.img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), color=(0, 0, 255), thickness=1)
                        text = str(label) + ': ' +'{:.2f}'.format(score)
                        # org
                        org = (int(pred_box[0])+5, max(0, int(pred_box[1]))-8)
                        # Using cv2.putText() method
                        cv2.putText(self.img, text, org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                        
                        
                        

            

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
            print("error in mmdetection")
        else:
            self.img_tmp = cv2_img
            
                
    def start(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                cv2.namedWindow('integrator_instance')        # Create a named window
                cv2.moveWindow('integrator_instance', 1500,600)  # Move it to (40,30)

                cv2.imshow('integrator_instance', self.img)
                
                
                key = cv2.waitKey(1)
                
            self.loop_rate.sleep()


def main():
    rospy.init_node('tas_integrator_o', anonymous=True)

    Ctas_object_subscriber_YJ = TAS_Object_Subscriber_YJ()

    Ctas_object_subscriber_YJ.start()


if __name__ == '__main__':
    main()


