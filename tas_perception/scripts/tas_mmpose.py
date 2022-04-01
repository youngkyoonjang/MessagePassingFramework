import os
from argparse import ArgumentParser

import cv2
from PIL import Image

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


import sys
sys.argv[0]="demo/top_down_img_demo_with_mmdet.py"

# sys.argv[1:]=["./models/hrnet_w48_coco_256x192.py","https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
# sys.argv[1:]=["./models/hrnet_w48_coco_256x192.py","./models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
sys.argv[1:]=["/ros-ws/src/tas_perception/models/hrnet_w48_coco_256x192.py","/ros-ws/src/tas_perception/models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"]
            #   "--runtime_img_in", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input.png",
            #   "--runtime_img_status", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input_status.txt",
            #   "--runtime_body_json_out", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/body_out.json"]

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


from matplotlib import pyplot as plt 


import os

import message_filters
from sensor_msgs.msg import Image, CameraInfo


class MMPose_YJ:
    def __init__(self, args):
        # params

        self.img = None
        self.instance_json = None
        self.jsonString = None
        self.pose_results = None
        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()

        assert args.show or (args.out_video_root != '')
        # assert args.det_config is not None
        # assert args.det_checkpoint is not None

        # det_model = init_detector(
        #     args.det_config, args.det_checkpoint, device=args.device.lower())
        # build the pose model from a config file and a checkpoint file
        print(args.pose_config)
        self.pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower())

        self.dataset = self.pose_model.cfg.data['test']['type']

        # args.video_path = args.test_video_in

        # self.scale_percent = 100
        # self.fps = 30.0

        self.return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        self.output_layer_names = None

        # self.det_cat_id = args.det_cat_id
        self.bbox_thr = args.bbox_thr
        self.kpt_thr = args.kpt_thr
        self.radius = args.radius
        self.thickness_ = args.thickness
        

        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"

        json_topic_instance = "/instance_json_out"
        rospy.Subscriber(json_topic_instance, String, self.instance_json_callback, (json_topic_instance))
        

        color_sub = message_filters.Subscriber(image_topic, Image)
        depth_sub = message_filters.Subscriber(aligned_depth_to_color, Image)
        message_filters.Subscriber(json_topic_instance, String)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)
        # ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, instance_sub], queue_size=10, slop=3, allow_headerless=True)

        self.pub = rospy.Publisher('body_json_out', String, queue_size=10)

        ts.registerCallback(self.image_callback)


    def instance_json_callback(self, msg, args):
        global instance_json
        json_topic = args[0]

        # print("Received an json!")
        try:
            # Convert your ROS Image message to OpenCV2
            loaded_dictionary = json.loads(msg.data)
        except CvBridgeError:
            print("e")
        else:
            self.instance_json = loaded_dictionary


    # from dateutil.parser import parse
    # def image_callback(msg1, msg2, msg3):
    def image_callback(self, msg1, msg2):
        rs_depth_scale = 0.0010000000474974513
        
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

            # self.instance_json = json.loads(msg3.data)
            integrated_data_tmp_for_saving=[]
            # integrated_data_tmp_for_saving = body_json[0]
            # integrated_data_tmp_for_saving['instances']=self.instance_json[0]['instances']

            if(self.instance_json!=None):
                
                if(0<len(self.instance_json[0]['instances'])):                
                    integrated_data_tmp_for_saving=self.instance_json[0]
                    integrated_data = [integrated_data_tmp_for_saving]
                    # body = integrated_data[0]['body']
                    instances_results = integrated_data[0]['instances']
                    # action = integrated_data[0]['action']
                    # video_instances = integrated_data[0]['video_instances']
                    # hand_states = integrated_data[0]['hand_states']


        except CvBridgeError:
            print("error in mmdetection")
        else:
            self.img = cv2_img

            if(self.instance_json!=None):
                
                if(0<len(self.instance_json[0]['instances'])):
                    
                    for i in range(0, len(instances_results['labels'])):
                        label = instances_results['labels'][i]
                        
                        if(label == 'person'):
                            # person_results = instances_results['pred_boxes'][i] #float
                            depth_results = instances_results['depth_info'][i] #float
                            bbox_info = [instances_results['pred_boxes'][i][0], instances_results['pred_boxes'][i][1],
                                        instances_results['pred_boxes'][i][2], instances_results['pred_boxes'][i][3],
                                        instances_results['scores'][i]]
                            bbox_np  = np.asarray(bbox_info)
                            person_results = [{'bbox': bbox_np}]
                            
                            # test a single image, with a list of bboxes.
                            self.pose_results, returned_outputs = inference_top_down_pose_model(
                                self.pose_model,
                                self.img,
                                person_results,
                                bbox_thr=self.bbox_thr,
                                format='xyxy',
                                dataset=self.dataset,
                                return_heatmap=self.return_heatmap,
                                outputs=self.output_layer_names)

                            #########################################
                            ## yjang inserted for writing results 
                            # td = timedelta(seconds=(i_cnt / video_fps))
                            
                            t = time.localtime()
                            td = time.strftime("%H:%M:%S", t)
                            if(0<len(self.pose_results)):
                                bodies = []
                                for i_tmp in range(0, len(self.pose_results)):
                                    pt = self.pose_results[i_tmp]['bbox'].tolist()
                                    keypoints = self.pose_results[i_tmp]['keypoints'].tolist()
                                    bodies.append({'bbox':pt, 'keypoints':keypoints})
                                
                                tmp_dic_ = {"frame_id": 0,
                                            "timestamp": str(td),
                                            "body":bodies}
                                    
                            else:
                                tmp_dic_ = {"frame_id": 0,
                                            "timestamp": str(td),
                                            "body":[]}
                            
                            # out_dic_list.append(tmp_dic_)
                            out_dic_list= [tmp_dic_]
                            #########################################



                            
                            #########################################
                            ## yjang inserted for writing results 
                            self.jsonString = json.dumps(out_dic_list, indent=4)
                            # jsonFile = open(out_info_file_path, "w")
                            # jsonFile.write(self.jsonString)
                            # # print(self.jsonString)
                            # # loaded_dictionary = json.loads(self.jsonString)
                            # jsonFile.close()
                            #########################################

    
    def start(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                # self.pub_img.publish(self.br.cv2_to_imgmsg(self.image, encoding="rgb8"))

                # cv2.namedWindow('Body', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Body')        # Create a named window
                cv2.moveWindow('Body', 10,100)  # Move it to (40,30)
                # show the results
                if self.pose_results is not None:
                    self.img = vis_pose_result(
                        self.pose_model,
                        self.img,
                        self.pose_results,
                        dataset=self.dataset,
                        kpt_score_thr=self.kpt_thr,
                        radius=self.radius,
                        thickness=self.thickness_,
                        show=False)

                cv2.imshow('Body', self.img)
                # plt.figure(1); plt.clf()
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                # plt.imshow(self.img)
                # plt.pause(0.03)
                
                # cv2.imwrite('./body_results_tmp.png', self.img)
                key = cv2.waitKey(1)
                # print('body-done')
                # cv2.destroyAllWindows()
            
            if self.jsonString is not None:
                # self.pub_coord.publish(self.obj_3d_coord)
                self.pub.publish(self.jsonString)
            
            self.loop_rate.sleep()





def main():

    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('det_config', help='Config file for detection')
    # parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    # parser.add_argument('--test_video_in', default="/media/DataDisk/yj_bitbucket/dev/mmpose/vis_results/robot_cooking_videos/short_sample_clip.mp4", type=str)
    # parser.add_argument('--mmpose_body_json_out', default="/media/DataDisk/yj_bitbucket/dev/mmpose/info_results/body_out.json", type=str)

    # # parser.add_argument('--runtime_img_in', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input.png", type=str)
    # # parser.add_argument('--runtime_body_json_out', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/body_out.json", type=str)
    # # parser.add_argument('--runtime_img_status', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input_status.txt", type=str)


    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    rospy.init_node('image_listener1', anonymous=True)

    CmmPose_YJ = MMPose_YJ(args)

    CmmPose_YJ.start()


if __name__ == '__main__':
    main()


