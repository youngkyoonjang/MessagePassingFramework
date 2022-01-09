## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

import argparse

import cv2
import torch

import numpy as np

from mmdet.apis import inference_detector, init_detector


import json
from datetime import timedelta
from datetime import datetime

################################################################################
############################### yjang inserted
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# Instantiate CvBridge


import json


import sys
'''
## demo with a bit of config file modification 
sys.argv[0]="tools/run_net.py"
sys.argv[1:]=["--cfg",video_root + "/dev/slowfast/configs/SSv2/SLOWFAST_16x8_R50_multigrid.yaml"]
'''
video_root = "/media/DataDisk/yj_bitbucket/dev"

## testing (YOLACT 101 is the best at the moment)
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/yolact/yolact_r50_1x8_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/yolact_r50_1x8_coco_20200908-f38d58df.pth",

# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/yolact/yolact_r101_1x8_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/yolact_r101_1x8_coco_20200908-4cbe9101.pth",


# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth",

# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth",

# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",

# "/media/DataDisk/yj_bitbucket/dev/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py",
# "/media/DataDisk/yj_bitbucket/dev/mmdetection/models/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth",

sys.argv[0]="demo/webcam_demo.py"
sys.argv[1:]=["/ros-ws/src/tas_perception/models/yolact_r101_1x8_coco.py",
"/ros-ws/src/tas_perception/models/yolact_r101_1x8_coco_20200908-4cbe9101.pth"]

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')

    parser.add_argument('--runtime_instance_json_out', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/instance_out.json", type=str)

    args = parser.parse_args()
    return args

###########################################
# additionally added for running visualisation

import mmcv
from mmdet.core.visualization import imshow_det_bboxes
from matplotlib.patches import Polygon

from dateutil.parser import parse

class MMDetection_YJ:
    def __init__(self, args):
        # params
        self.camera_to_kinova_tf = [[+0.06222415, +0.13734534, +0.98856685, -0.19234757],
                                    [-0.99604155, +0.07154102, +0.05275517, +0.08416595],
                                    [-0.06347741, -0.98793630, +0.14125325, +0.33030282],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]]
        
        self.CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

        self.img = None
        self.images = None
        self.jsonString = None

        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

        # Instantiate CvBridge
        self.bridge = CvBridge()


        device = torch.device(args.device)
        # device = torch.device("cuda:1")
        
        self.model = init_detector(args.config, args.checkpoint, device=device)

        # Define your image topic
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        aligned_depth_to_color = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = 'camera/color/camera_info'
        
        self.down_sizing_factor = 0.5
        color_sub = message_filters.Subscriber(image_topic, Image)
        depth_sub = message_filters.Subscriber(aligned_depth_to_color, Image)

        ## CameraInfo: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        ## rs2_intrinsics: https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/h/rs_types.h#L55
        ## The Pinhole Camera Matrix: https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/CV/PinholeCamera/PinholeCamera.html
        ## convert_depth_pixel_to_metric_coordinate: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py
        self.camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.camera_intrinsics = {'ppx':self.camera_info.K[2], 'ppy':self.camera_info.K[5], 'fx': self.camera_info.K[0], 'fy': self.camera_info.K[4]} # fx, fy, cz, cy
        
        # print(self.camera_info.K)
        # print(self.camera_intrinsics)

        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=10)

        self.pub = rospy.Publisher('instance_json_out', String, queue_size=10)
        ts.registerCallback(self.image_callback)

        self.get_homogeneous_transformation_matrix()

        self.random_rotation=[0,0,0]

    def convert_depth_pixel_to_metric_coordinate(self, depth, pixel_x, pixel_y, camera_intrinsics):
        """
        Convert the depth and image point information to metric coordinates
        Parameters:
        -----------
        depth 	 	 	 : double
                            The depth value of the image point
        pixel_x 	  	 	 : double
                            The x value of the image coordinate
        pixel_y 	  	 	 : double
                                The y value of the image coordinate
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        Return:
        ----------
        X : double
            The x value in meters
        Y : double
            The y value in meters
        Z : double
            The z value in meters
        """
        # X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
        # Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
        X = (pixel_x/self.down_sizing_factor - camera_intrinsics['ppx'])/camera_intrinsics['fx'] *depth
        Y = (pixel_y/self.down_sizing_factor - camera_intrinsics['ppy'])/camera_intrinsics['fy'] *depth

        # print(pixel_x/self.down_sizing_factor, pixel_y/self.down_sizing_factor, X, Y)
        return X, Y, depth

    def locate_point_in_given_local_coordinate(self, pt_3d, homogeneous_transformation_mat):        
        xyz_in_given_coordinate = np.matmul(homogeneous_transformation_mat, [pt_3d[0], pt_3d[1], pt_3d[2], 1])

        return xyz_in_given_coordinate[0], xyz_in_given_coordinate[1], xyz_in_given_coordinate[2]

    def convert_metric_coordinate_to_depth_pixel(self, pt_3d, camera_intrinsics):        
        X = pt_3d[0]
        Y = pt_3d[1]
        Z = pt_3d[2]

        u = ((X/Z) * camera_intrinsics['fx']) + camera_intrinsics['ppx']
        v = ((Y/Z) * camera_intrinsics['fy']) + camera_intrinsics['ppy']

        # print(pixel_x/self.down_sizing_factor, pixel_y/self.down_sizing_factor, X, Y)
        return u*self.down_sizing_factor,v*self.down_sizing_factor
    
    def get_homogeneous_transformation_matrix(self, rotation_vec=[0,0,0], translation_vec=[0,0,0]):
        TransMat = [[1,0,0,translation_vec[0]],
                    [0,1,0,translation_vec[1]],
                    [0,0,1,translation_vec[2]],
                    [0,0,0,1]]

        a = rotation_vec[2] # z-axis
        b = rotation_vec[1] # y-axis
        r = rotation_vec[0] # x-axis
        yaw = [[np.cos(a), -np.sin(a), 0],
               [np.sin(a),  np.cos(a), 0],
               [0,          0,         1]]
        pitch = [[np.cos(b), 0, np.sin(b)],
                 [0,         1, 0],
                 [-np.sin(b), 0, np.cos(b)]]
        roll = [[1, 0,          0],
                [0, np.cos(r), -np.sin(r)],                
                [0, np.sin(r),  np.cos(r)]]

        RotMat = np.matmul(np.matmul(yaw,pitch), roll)
        RotMat_expanded = [[RotMat[0][0],RotMat[0][1],RotMat[0][2],0],
                           [RotMat[1][0],RotMat[1][1],RotMat[1][2],0],
                           [RotMat[2][0],RotMat[2][1],RotMat[2][2],0],
                           [0,        0,        0,        1]]

        HTM = np.matmul(TransMat, RotMat_expanded)

        return HTM

    def color_val_matplotlib(self, color):
        """Convert various input in BGR order to normalized RGB matplotlib color
        tuples,

        Args:
            color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

        Returns:
            tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
        """
        color = mmcv.color_val(color)
        color = [color / 255 for color in color[::-1]]
        return tuple(color)

    # from dateutil.parser import parse
    def image_callback(self, msg1, msg2):
        rs_depth_scale = 0.0010000000474974513
        
        try:
            # Convert your ROS Image message to OpenCV2 for color
            color_img = self.bridge.imgmsg_to_cv2(msg1, "bgr8")
            cv2_img = cv2.resize(color_img, (int(color_img.shape[1]*self.down_sizing_factor), int(color_img.shape[0]*self.down_sizing_factor)))

            
            # Convert your ROS Image message to OpenCV2 for depth
            depth_image = self.bridge.imgmsg_to_cv2(msg2, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)

            # grey_color = 153
            depth_image_3d = np.dstack((depth_array,depth_array,depth_array)) #depth image is 1 channel, color is 3 channels
            cv2_depth_img = cv2.resize(depth_image_3d, (int(depth_image_3d.shape[1]*self.down_sizing_factor), int(depth_image_3d.shape[0]*self.down_sizing_factor)))
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth_img, alpha=0.03), cv2.COLORMAP_JET)

            depth_img = depth_colormap

        except CvBridgeError:
            print("error in mmdetection")
        else:
            self.img = cv2_img


            # Returns a datetime object containing the local date and time
            dateTimeObj = datetime.now()
            # get the time object from datetime object
            timeObj = dateTimeObj.time()


            result = inference_detector(self.model, self.img)
            ###########################################
            # brought this code from the above function
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            # draw segmentation masks
            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                else:
                    segms = np.stack(segms, axis=0)
            
            class_names=self.CLASSES
            score_thr=0.4
            bbox_color=(72, 101, 241)
            text_color=(72, 101, 241)
            mask_color=None
            # thickness=2
            # font_size=13
            # win_name=''
            # show=False
            # wait_time=0
            # out_file=None

            assert bboxes.ndim == 2, \
                f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
            assert labels.ndim == 1, \
                f' labels ndim should be 1, but its ndim is {labels.ndim}.'
            assert bboxes.shape[0] == labels.shape[0], \
                'bboxes.shape[0] and labels.shape[0] should have the same length.'
            assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
                f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
            self.img = mmcv.imread(self.img).astype(np.uint8)

            if score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]

            mask_colors = []
            if labels.shape[0] > 0:
                if mask_color is None:
                    # Get random state before set seed, and restore random state later.
                    # Prevent loss of randomness.
                    # See: https://github.com/open-mmlab/mmdetection/issues/5844
                    state = np.random.get_state()
                    # random color
                    np.random.seed(42)
                    mask_colors = [
                        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                        for _ in range(max(labels) + 1)
                    ]
                    np.random.set_state(state)
                else:
                    # specify  color
                    mask_colors = [
                        np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
                    ] * (
                        max(labels) + 1)

            bbox_color = self.color_val_matplotlib(bbox_color)
            text_color = self.color_val_matplotlib(text_color)

            self.img = mmcv.bgr2rgb(self.img)
            width, height = self.img.shape[1], self.img.shape[0]
            self.img = np.ascontiguousarray(self.img)

            ################################################################################
            ############################### yjang inserted
            instances_tmp_list = {}

            import time
            t = time.localtime()
            td = time.strftime("%H:%M:%S", t)

            boxes_ = bboxes[:,:4].tolist()
            scores_ = bboxes[:,-1].tolist()
            classes_ = labels.tolist()
            labels_ = []
            mask_contour_list_ = []
            depth_list_ = []
            xyz_ = []
            xyz_kinova_ = []
            ############################### yjang inserted
            ################################################################################
            


            polygons = []
            color = []
            centres_ = []
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                bbox_int = bbox.astype(np.int32)
                poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                        [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
                np_poly = np.array(poly).reshape((4, 2))
                polygons.append(Polygon(np_poly))
                color.append(bbox_color)
                label_text = class_names[
                    label] if class_names is not None else f'class {label}'

                labels_.append(label_text) ##yjang inserted

                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'

                color_ = (int(color[0][0]*255), int(color[0][1]*255), int(color[0][2]*255))
                # self.img = cv2.rectangle(self.img, (poly[0][0], poly[0][1]), (poly[2][0], poly[2][1]), color_, thickness)

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                # self.img = cv2.putText(self.img, label_text, (poly[0][0], poly[0][1]), font, 
                #     fontScale, color_, thickness, cv2.LINE_AA)
                preds=None

                feature_realtime_tmp = [-1,-1,-1,-1]
                xyz_tmp = [-1,-1,-1]
                xyz_kinova_tmp = [-1,-1,-1]
                if segms is not None:
                    color_mask = mask_colors[labels[i]]
                    mask = segms[i].astype(bool)
                    self.img[mask] = self.img[mask] * 0.5 + color_mask * 0.5

                    ################################################################################
                    ############################### yjang inserted
                    contours, hierarchy = cv2.findContours(segms[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    mask_contour_list_.append(contours[0].tolist())
                    masK_array = np.array(mask_contour_list_[0]).astype(np.int32)


                    ## calculating centre of mass of contour is from following pages
                    ## https://en.wikipedia.org/wiki/Image_moment
                    ## https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/                
                    # if(labels_[i] == 'person'):
                    M = cv2.moments(contours[0])
                    if (M["m00"]!=0): 
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        if(cX < 0): cX=0
                        elif(cv2_depth_img.shape[1] <= cX): cX=cv2_depth_img.shape[1]-1
                        if(cY < 0): cY=0
                        elif(cv2_depth_img.shape[0] <= cY): cY=cv2_depth_img.shape[0]-1

                        centre_depth = cv2_depth_img[cY, cX, 0]
                        val_in_meter = centre_depth * rs_depth_scale
                        
                        cv2.putText(self.img, "{:.4f}".format(val_in_meter)+"M", (cX - 5, cY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.circle(self.img, (cX, cY), 7, (255, 255, 255), -1)
                        # feature_realtime_tmp = [str(cX), str(cY), float(centre_depth), str(val_in_meter)]
                        feature_realtime_tmp = [cX, cY, float(centre_depth), val_in_meter]
                        # xyz_tmp = [self.convert_depth_pixel_to_metric_coordinate(val_in_meter, cX, cY, self.camera_intrinsics)]
                        xyz_tmp = [self.convert_depth_pixel_to_metric_coordinate(val_in_meter, cX, cY, self.camera_intrinsics)]
                        cv2.putText(self.img, "{:.4f}".format(xyz_tmp[0][0]), (cX - 5, cY - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(self.img, "{:.4f}".format(xyz_tmp[0][1]), (cX - 5, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # print(xyz_tmp[0])
                        
                        xyz_kinova_tmp = np.matmul(self.camera_to_kinova_tf, [xyz_tmp[0][0], xyz_tmp[0][1], xyz_tmp[0][2], 1])
                        xyz_kinova_tmp = xyz_kinova_tmp[:-1].tolist()
                        # print('kinova_:', xyz_kinova_tmp)

                        if(xyz_tmp[0][2]>0):
                            homogeneous_transformation_mat = self.get_homogeneous_transformation_matrix(rotation_vec=self.random_rotation, translation_vec=xyz_tmp[0])
                            # print(homogeneous_transformation_mat)
                            offset = 0.05
                            x_tmp, y_tmp, z_tmp = self.locate_point_in_given_local_coordinate([offset,0,0], homogeneous_transformation_mat)
                            u, v = self.convert_metric_coordinate_to_depth_pixel([x_tmp, y_tmp, z_tmp], self.camera_intrinsics)
                            cv2.line(self.img, (cX,cY), (int(u),int(v)), (255,0,0), 8)
                            x_tmp, y_tmp, z_tmp = self.locate_point_in_given_local_coordinate([0,offset,0], homogeneous_transformation_mat)
                            u, v = self.convert_metric_coordinate_to_depth_pixel([x_tmp, y_tmp, z_tmp], self.camera_intrinsics)
                            cv2.line(self.img, (cX,cY), (int(u),int(v)), (0,255,0), 8)
                            x_tmp, y_tmp, z_tmp = self.locate_point_in_given_local_coordinate([0,0,offset], homogeneous_transformation_mat)
                            u, v = self.convert_metric_coordinate_to_depth_pixel([x_tmp, y_tmp, z_tmp], self.camera_intrinsics)
                            cv2.line(self.img, (cX,cY), (int(u),int(v)), (0,0,255), 8)
                            
                            
                        # print(xyz_tmp)
                    ############################### yjang inserted
                    ################################################################################
                depth_list_.append(feature_realtime_tmp)
                xyz_.append(xyz_tmp[0])
                xyz_kinova_.append(xyz_kinova_tmp)
                            
            ################################################################################
            ############################### yjang inserted
            if(0<len(boxes_)):
                instances_tmp_list["pred_boxes"] = boxes_
                instances_tmp_list["scores"] = scores_
                instances_tmp_list["pred_classes"] = classes_
                instances_tmp_list["labels"] = labels_
                instances_tmp_list["pred_contours"] = mask_contour_list_
                instances_tmp_list["depth_info"] = depth_list_
                instances_tmp_list["xyz"] = xyz_
                instances_tmp_list["xyz_kinova"] = xyz_kinova_

                tmp_dic_ = {"frame_id": 0,
                            "timestamp": str(td),
                            "instances":instances_tmp_list}
            else:
                tmp_dic_ = {"frame_id": 0,
                            "timestamp": str(td),
                            "instances":[]}

            out_dic_list= [tmp_dic_]


            self.img = mmcv.rgb2bgr(self.img)


            # ####################################################################################
            # ## yjang inserted for centre depth acquisition
            # centre_x = int(depth_array.shape[1] / 2)
            # centre_y = int(depth_array.shape[0] *.5) #int(depth_image.shape[0] / 2)
            # centre_depth = depth_array[centre_y, centre_x]
            # cv2.circle(depth_img, (centre_x, centre_y), 10, (0,0,255), -1) ## yjang inserted
            # cv2.circle(self.img, (centre_x, centre_y), 10, (0,0,255), -1) ## yjang inserted
            # ## yjang inserted for centre depth acquisition
            # ####################################################################################
                    



            self.images = np.hstack((self.img, depth_colormap))

            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', self.images)

            # key = cv2.waitKey(10)


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
            # if(self.random_rotation[0]+0.01 < 360):
            #     self.random_rotation[0] = self.random_rotation[0]+0.01
            # else:
            #     self.random_rotation[0] = 0
            
            # if(self.random_rotation[1]+0.02 < 360):
            #     self.random_rotation[1] = self.random_rotation[1]+0.02
            # else:
            #     self.random_rotation[1] = 0
            
            # if(self.random_rotation[2]+0.03 < 360):
            #     self.random_rotation[2] = self.random_rotation[2]+0.03
            # else:
            #     self.random_rotation[2] = 0

            # if self.img is not None:
            if self.images is not None:
                # self.pub_img.publish(self.br.cv2_to_imgmsg(self.image, encoding="rgb8"))

                # cv2.namedWindow('Body', cv2.WINDOW_NORMAL)

                cv2.imshow('Instances', self.images)
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



import os

import message_filters
from sensor_msgs.msg import Image, CameraInfo

def main():

    args = parse_args()

    rospy.init_node('image_listener2', anonymous=True)

    CmmDetection_YJ = MMDetection_YJ(args)

    CmmDetection_YJ.start()


if __name__ == '__main__':
    main()


