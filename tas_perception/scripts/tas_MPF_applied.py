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
        
        ###################################################################################
        ###################################################################################
        ## instance (object) subscriber initialisation
        ###################################################################################       
        self.instance_json = None
        
        json_topic_instance = "/instance_json_out"
        rospy.Subscriber(json_topic_instance, String, self.instance_json_callback)
        
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
        
        ###################################################################################
        ###################################################################################
        ## pose (body) subscriber initialisation
        ###################################################################################
        self.body_json = None
        
        json_topic_body = "/body_json_out"
        rospy.Subscriber(json_topic_body, String, self.body_json_callback)
        
        # self.det_cat_id = args.det_cat_id
        self.bbox_thr = 0.3
        self.kpt_thr = 0.3
        self.radius = 4
        self.thickness_ = 1
        
        
        ###################################################################################
        ###################################################################################
        ## hand subscriber initialisation
        ###################################################################################
        self.hand_json = None
        
        json_topic_hand = "/hand_state_json_out"
        rospy.Subscriber(json_topic_hand, String, self.hand_json_callback)
        
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
        
        
        ###################################################################################
        ###################################################################################
        ## integrated data initialisation
        ###################################################################################
        self.integrated_data=[ { 'frame_id':0,
                            'timestamp':'0:00:00',
                            'body':[],
                            'instances':[],
                            'hand_states': {'hands':[],
                                            'objects':[]
                                           }
                          } ]
        
        self.object_count_list = []
        self.bVisualising_Memory = True
        self.original_list=[]
        
        self.furniture_candidates=['sink', 'desk', 'dining_table', 'bed'] ## set object list that could be a furniture
        self.target_object_list=['cup', 'mouse', 'knife', 'keyboard', 'laptop'] ## set object list that you plan to interact 
        
        self.iou_threshold = 0.2
     
        ###################################################################################
        ###################################################################################
        ## common stuff initialisation
        ###################################################################################
        self.img = None
        self.img_tmp = None
        
        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()
        
        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"
        
        color_sub = message_filters.Subscriber(image_topic, Image)
        depth_sub = message_filters.Subscriber(aligned_depth_to_color, Image)
        # message_filters.Subscriber(json_topic_hand, String)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)
        
        # self.pub = rospy.Publisher('memory_json_out', String, queue_size=10)

        ts.registerCallback(self.image_callback)

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def update_object_list (self, original_list, instances, furniture_candidates):
        if(len(original_list)==0 and 0<len(instances['pred_boxes'])):
            for i in range(0, len(instances['pred_boxes'])):
                bbox = instances['pred_boxes'][i]
                label = instances['labels'][i]
                score = instances['scores'][i]

                instances_tmp = {'bbox':bbox, 'label':{label:1}, 'representative_label':label, 'score':{label:score}, 'representative_score':score}
                original_list.append(instances_tmp)
        else:
            for i in range(0, len(instances['pred_boxes'])):
                bbox = instances['pred_boxes'][i]
                label = instances['labels'][i]
                score = instances['scores'][i]

                max_IoU = -1
                max_IoU_id = -1
                for j in range(0, len(original_list)):
                    org_bbox = original_list[j]['bbox']
                    iou = self.bb_intersection_over_union(org_bbox, bbox)

                    if(max_IoU < iou and self.iou_threshold < iou):
                        max_IoU = iou
                        max_IoU_id = j
                        max_IoU_label = instances['labels'][i]

                if(max_IoU_id != -1):
                    if(label in original_list[max_IoU_id]['label']):
                        original_list[max_IoU_id]['label'][label]+=1
                        if(original_list[max_IoU_id]['score'][label]<score):
                            original_list[max_IoU_id]['score'][label]=score
                    else:

                        if(label=='person'):
                            print('here')
                        original_list[max_IoU_id]['label'][label]=1
                        original_list[max_IoU_id]['score'][label]=score
                    original_list[max_IoU_id]['bbox']=bbox
                    
                    # max_count_key = max(original_list[max_IoU_id]['label'], key=original_list[max_IoU_id]['label'].get)
                    # max_score_key = max(original_list[max_IoU_id]['score'], key=original_list[max_IoU_id]['score'].get)

                    def normalize(d, target=1.0):
                        raw = sum(d.values())
                        factor = target/raw
                        return {key:value*factor for key,value in d.items()}

                    def weighted_sum(count_dic, score_dic):
                        return {key:(value*0.5 + score_dic[key]*0.5) for key,value in count_dic.items()}

                    normalised_label_dic = normalize(original_list[max_IoU_id]['label'])
                    weighted_sum_dic = weighted_sum(normalised_label_dic, original_list[max_IoU_id]['score'])
                    max_weighted_sum_key = max(weighted_sum_dic, key=weighted_sum_dic.get)
                            
                    original_list[max_IoU_id]['representative_label']=max_weighted_sum_key
                    original_list[max_IoU_id]['representative_score']=original_list[max_IoU_id]['score'][max_weighted_sum_key]
                else:
                    instances_tmp = {'bbox':bbox, 'label':{label:1}, 'representative_label':label, 'score':{label:score}, 'representative_score':score}
                    original_list.append(instances_tmp)
                    #print ("new item:", label, score)

        furniture_id_in_memory = []
        for j in range(0, len(original_list)):
            representative_label = original_list[j]['representative_label']
            for furniture_tmp in furniture_candidates:    
                if (furniture_tmp in representative_label):
                    furniture_id_in_memory.append(j)
                    break
                    
        return original_list, furniture_id_in_memory

    def filtering_based_on_grativy(self, original_list, Contacted_object_bbox, furniture_id_in_memory):
        for f_i in furniture_id_in_memory:
            f_bbox = original_list[f_i]['bbox']
            for i in range(0, len(original_list)):
                if(i!=f_i and original_list[i] != None):
                    bbox = original_list[i]['bbox']
                    iou = self.bb_intersection_over_union(f_bbox, bbox)
                    iou_hand = self.bb_intersection_over_union(Contacted_object_bbox, bbox)

                    if(iou==0 and iou_hand==0):
                        original_list[i] = None

        for i in range(len(original_list), 0, -1):
            id = i-1
            if(original_list[id] == None):
                original_list.pop(id)
                
        return original_list

    def filtering_based_on_redundancy(self, original_list, instances):
        object_list_temp = {}
        for j in range(0, len(original_list)):
            org_bbox = original_list[j]['bbox']        
            representative_label = original_list[j]['representative_label']
            if(representative_label not in object_list_temp):
                object_list_temp[representative_label]=[j]
            else:
                object_list_temp[representative_label].append(j)
                
        for i in range(0, len(instances['pred_boxes'])):
            bbox = instances['pred_boxes'][i]
            label = instances['labels'][i]
            score = instances['scores'][i]
            
            max_IoU = -1
            max_IoU_id = -1
            
            object_list_temp = {}
            for j in range(0, len(original_list)):
                org_bbox = original_list[j]['bbox']
                iou = self.bb_intersection_over_union(org_bbox, bbox)
                
                if(max_IoU < iou and self.iou_threshold < iou):
                    max_IoU = iou
                    max_IoU_id = j
                    max_IoU_label = instances['labels'][i] 
                    
                representative_label = original_list[j]['representative_label']
                if(representative_label not in object_list_temp):
                    object_list_temp[representative_label]=[j]
                else:
                    object_list_temp[representative_label].append(j)
                    
            if(max_IoU_id != -1):
                keys = [k for k, v in object_list_temp.items() if max_IoU_id in v]
                for id in object_list_temp[keys[0]]:
                    if(id != max_IoU_id):
                        original_list[id] = None
                        
                for i in range(len(original_list), 0, -1):
                    id = i-1
                    if(original_list[id] == None):
                        original_list.pop(id)
                        
        return original_list

    def instance_json_callback(self, msg):
        try:
            loaded_dictionary_ins = json.loads(msg.data)
            self.instance_json = loaded_dictionary_ins
        except CvBridgeError:
            print("e")
        else:
            if(0<len(self.instance_json[0]['instances'])):
                self.integrated_data[0]['instances']=self.instance_json[0]['instances']
            else:
                self.integrated_data[0]['instances']=[]
                        

    def body_json_callback(self, msg):
        try:
            loaded_dictionary_body = json.loads(msg.data)
            self.body_json = loaded_dictionary_body
        except CvBridgeError:
            print("e")
        else:
            if(0<len(self.body_json[0]['body'])):
                self.integrated_data[0]['body']=self.body_json[0]['body']
            else:
                self.integrated_data[0]['body']=[]
            
    def hand_json_callback(self, msg):
        try:
            loaded_dictionary_hand = json.loads(msg.data)

            self.hand_json = loaded_dictionary_hand
        except CvBridgeError:
            print("e")
        else:
            if(0<len(self.hand_json[0]['hands'])):
                self.integrated_data[0]['hand_states']['hands']=self.hand_json[0]['hands']
            else:
                self.integrated_data[0]['hand_states']['hands']=[]
                
            if(0<len(self.hand_json[0]['objects'])):
                self.integrated_data[0]['hand_states']['objects']=self.hand_json[0]['objects']
            else:
                self.integrated_data[0]['hand_states']['objects']=[]
                
                
                    
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
            if self.img_tmp is not None:
                self.img = self.img_tmp
                
                if(self.hand_json!=None):                
                    hand_states = self.integrated_data[0]['hand_states']                    
                        
                if(self.body_json!=None):
                    body = self.integrated_data[0]['body']
                    
                            
                if(self.instance_json!=None):
                    instances = self.integrated_data[0]['instances']

                ##########################################################
                # instance status update
                if(self.hand_json!=None and self.body_json!=None and self.instance_json!=None):
                    if(0<len(instances)):
                        self.original_list, furniture_id_in_memory = self.update_object_list (self.original_list, instances, self.furniture_candidates)
                    
                        if(0<len(body)):
                            hand_pt_tmp = body[0]['keypoints'][10]
                            hand_bbox_tmp = [int(hand_pt_tmp[0]-20), int(hand_pt_tmp[1]), int(hand_pt_tmp[0]+40), int(hand_pt_tmp[1]+60)]
                            #cv2.rectangle(img_test_in, (int(hand_bbox_tmp[0]), int(hand_bbox_tmp[1])), (int(hand_bbox_tmp[2]), int(hand_bbox_tmp[3])), (255,0,255), 1, lineType=cv2.LINE_8)
                            self.original_list = self.filtering_based_on_grativy(self.original_list, hand_bbox_tmp, furniture_id_in_memory)
                            self.original_list = self.filtering_based_on_redundancy(self.original_list, instances)

                    IsHandContacting_portable_object = False
                    Contacted_object_bbox = []
                    for i in range(0, len(hand_states['hands'])):
                        if(hand_states['hands'][i]['state'] == 3 and hand_states['hands'][i]['side_id'] == 1):
                            IsHandContacting_portable_object = True
                            # Contacted_object_bbox = hand_states['objects'][0]['bbox']
                            Contacted_object_bbox = hand_states['hands'][i]['bbox']
                            break
                    
                    if(IsHandContacting_portable_object):
                        max_IoU = 0
                        max_IoU_id = -1
                        max_IoU_label = ""
                        for i in range(0, len(self.original_list)):
                            bbox = self.original_list[i]['bbox']
                            label = self.original_list[i]['representative_label']
                            
                            if (not (label in self.target_object_list)):
                                continue
                            
                            iou = self.bb_intersection_over_union(Contacted_object_bbox,bbox )
                            print(label, self.original_list[i]['representative_label'], iou)

                            # if(max_IoU < iou and self.iou_threshold < iou):
                            if(max_IoU < iou):
                                max_IoU = iou
                                max_IoU_id = i
                                max_IoU_label = self.original_list[i]['representative_label']
                                max_IoU_label = self.original_list[i]['representative_label']

                        # if(max_IoU_id==-1):
                        #     max_IoU_label = "knife"
                        if(max_IoU_id!=-1):
                            self.original_list[max_IoU_id]['bbox']=Contacted_object_bbox
                            
                            if(self.bVisualising_Memory):
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                for i in range(0, len(self.original_list)):
                                    bbox_tmp = self.original_list[i]['bbox']
                                    label_tmp = self.original_list[i]['representative_label']
                                    cv2.rectangle(self.img, (int(bbox_tmp[0]), int(bbox_tmp[1])), (int(bbox_tmp[2]), int(bbox_tmp[3])), (0,0,255), 1, lineType=cv2.LINE_8)
                                    cv2.putText(self.img,label_tmp,(int(bbox_tmp[0]),int(bbox_tmp[1])-10), font, 0.8,(10,10,255),2)

                                cv2.rectangle(self.img, (int(Contacted_object_bbox[0]), int(Contacted_object_bbox[1])), (int(Contacted_object_bbox[2]), int(Contacted_object_bbox[3])), (255,255,255), 1, lineType=cv2.LINE_8)

                            print("contacted_item:", max_IoU_label, max_IoU)
                            output_str = "Grab,"+max_IoU_label

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            str_ = "Touched_object: "+max_IoU_label
                            #cv2.putText(img_test_in,str_,(50,200), font, 1,(105,0,0),2)
                    else:
                        print("contacted_item:", "None", "None")
                        output_str = "NoGrab,None"
                    
                            
            if self.img is not None:
                cv2.namedWindow('integrator_hand_pose_object')        # Create a named window
                cv2.moveWindow('integrator_hand_pose_object', 1500,600)  # Move it to (40,30)

                cv2.imshow('integrator_hand_pose_object', self.img)

                key = cv2.waitKey(1)

           
            self.loop_rate.sleep()


def main():
    rospy.init_node('tas_integrator_hpo', anonymous=True)

    Ctas_Integrator_YJ = TAS_Integrator_YJ()

    Ctas_Integrator_YJ.start()


if __name__ == '__main__':
    main()


