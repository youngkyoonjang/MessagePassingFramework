#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

script_path = os.path.dirname(os.path.realpath(__file__))

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

import os

import message_filters
from sensor_msgs.msg import Image, CameraInfo

import json
############################### yjang inserted
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.abspath(os.path.join(script_path, '/rt_gene/rt_gene_standalone/samples_gaze/')),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--save-estimate', dest='save_estimate', action='store_true', help='Save the predictions in a text file')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--gaze_backend', choices=['tensorflow', 'pytorch'], default='tensorflow')
    parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(script_path, '/rt_gene/rt_gene_standalone/samples_gaze/out')),
                        help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '/rt_gene/rt_gene/model_nets/Model_allsubjects1.h5'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-facedetection', dest="device_id_facedetection", type=str, default='cuda:0', help='Pytorch device id. Set to "cpu:0" to disable cuda')

    parser.set_defaults(vis_gaze=True)
    parser.set_defaults(save_gaze=True)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=True)
    parser.set_defaults(save_estimate=True)

    args = parser.parse_args()

    return args



class RT_GENE_GAZE_YJ:
    def __init__(self, args):
        # params
        self.args = args


        tqdm.write('Loading networks')
        self.landmark_estimator = LandmarkMethodBase(device_id_facedetection=self.args.device_id_facedetection,
                                                checkpoint_path_face=os.path.abspath(os.path.join(script_path, "/rt_gene/rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                                checkpoint_path_landmark=os.path.abspath(
                                                    os.path.join(script_path, "/rt_gene/rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                                model_points_file=os.path.abspath(os.path.join(script_path, "/rt_gene/rt_gene/model_nets/face_model_68.txt")))
        
        if self.args.gaze_backend == "tensorflow":
            from rt_gene.estimate_gaze_tensorflow import GazeEstimator

            self.gaze_estimator = GazeEstimator("/gpu:0", self.args.models)
        elif self.args.gaze_backend == "pytorch":
            from rt_gene.estimate_gaze_pytorch import GazeEstimator

            self.gaze_estimator = GazeEstimator("cuda:0", self.args.models)
        else:
            raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

        self.head_pose_image = None
        self.s_gaze_img = None
        
        self.img = None
        self.images = None
        self.jsonString = None

        self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz
        # Instantiate CvBridge
        self.bridge = CvBridge()

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

        self.pub = rospy.Publisher('gaze_json_out', String, queue_size=10)
        ts.registerCallback(self.image_callback)



    def load_camera_calibration(self, calibration_file):
        import yaml
        with open(calibration_file, 'r') as f:
            cal = yaml.safe_load(f)

        dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
        camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

        return dist_coefficients, camera_matrix


    def extract_eye_image_patches(self, subjects):
        for subject in subjects:
            le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, self.landmark_estimator.eye_image_size)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c


    def estimate_gaze(self, base_name, color_img, dist_coefficients, camera_matrix):
        faceboxes = self.landmark_estimator.get_face_bb(color_img)
        if len(faceboxes) == 0:
            tqdm.write('Could not find faces in the image')
            return
        
        subjects = self.landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
        self.extract_eye_image_patches(subjects)
        
        input_r_list = []
        input_l_list = []
        input_head_list = []
        valid_subject_list = []

        ################################################################################
        ############################### yjang inserted
        gaze_tmp_list = {}
        import time
        t = time.localtime()
        td = time.strftime("%H:%M:%S", t)
        
        subject_id_ = []
        headpose_ = []
        gaze_ = []

        for idx, subject in enumerate(subjects):
            if subject.left_eye_color is None or subject.right_eye_color is None:
                tqdm.write('Failed to extract eye image patches')
                continue

            success, rotation_vector, _ = cv2.solvePnP(self.landmark_estimator.model_points,
                                                    subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                    cameraMatrix=camera_matrix,
                                                    distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

            
            if not success:
                tqdm.write('Not able to extract head pose for subject {}'.format(idx))
                continue

            _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
            _m = np.zeros((4, 4))
            _m[:3, :3] = _rotation_matrix
            _m[3, 3] = 1
            # Go from camera space to ROS space
            _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]
            roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
            roll_pitch_yaw = limit_yaw(roll_pitch_yaw)
            phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

            face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            self.head_pose_image = self.landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))
            
            # if self.args.vis_headpose:
            #     plt.axis("off")
            #     plt.imshow(cv2.cvtColor(self.head_pose_image, cv2.COLOR_BGR2RGB))
            #     plt.show()

            # if self.args.save_headpose:
            #     # add idx to cope with multiple persons in one image
            #     # cv2.imwrite(os.path.join(self.args.output_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), self.head_pose_image)
            #     cv2.imshow("headpose", self.head_pose_image)
            #     cv2.waitKey(1)
            
            input_r_list.append(self.gaze_estimator.input_from_image(subject.right_eye_color))
            input_l_list.append(self.gaze_estimator.input_from_image(subject.left_eye_color))
            input_head_list.append([theta_head, phi_head])
            valid_subject_list.append(idx)
        
        if len(valid_subject_list) == 0:
            return
        
        gaze_est = self.gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                        inference_input_right_list=input_r_list,
                                                        inference_headpose_list=input_head_list)
        
        for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
            subject = subjects[subject_id]
            # Build visualizations
            r_gaze_img = self.gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
            l_gaze_img = self.gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
            self.s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)
            # if self.args.vis_gaze:
            #     plt.axis("off")
            #     plt.imshow(cv2.cvtColor(self.s_gaze_img, cv2.COLOR_BGR2RGB))
            #     plt.show()

            # if self.args.save_gaze:
            #     # add subject_id to cope with multiple persons in one image
            #     # cv2.imwrite(os.path.join(self.args.output_path, os.path.splitext(base_name)[0] + '_gaze_%s.jpg'%(subject_id)), self.s_gaze_img)
            #     # cv2.imwrite(os.path.join(self.args.output_path, os.path.splitext(base_name)[0] + '_left.jpg'), subject.left_eye_color)
            #     # cv2.imwrite(os.path.join(self.args.output_path, os.path.splitext(base_name)[0] + '_right.jpg'), subject.right_eye_color)
            #     cv2.imshow("gaze", self.s_gaze_img)
            #     cv2.waitKey(1)
            
            #if self.args.save_estimate:
            #     # add subject_id to cope with multiple persons in one image
            #     with open(os.path.join(self.args.output_path, os.path.splitext(base_name)[0] + '_output_%s.txt'%(subject_id)), 'w+') as f:
            #         f.write(os.path.splitext(base_name)[0] + ', [' + str(headpose[1]) + ', ' + str(headpose[0]) + ']' +
            #                 ', [' + str(gaze[1]) + ', ' + str(gaze[0]) + ']' + '\n')
            #    
            #    print(base_name + ', [' + str(headpose[1]) + ', ' + str(headpose[0]) + ']' + ', [' + str(gaze[1]) + ', ' + str(gaze[0]) + ']' + '\n')
            #    print('\n\n\n\n')
            
            subject_id_.append(subject_id)
            headpose_.append(headpose)
            gaze_.append(gaze)

        if(0<len(subject_id_)):
            gaze_tmp_list["subject_id"] = subject_id_
            gaze_tmp_list["headpose"] = headpose_
            gaze_tmp_list["gaze"] = gaze_

            tmp_dic_ = {"frame_id": 0,
                        "timestamp": str(td),
                        "gaze":gaze_tmp_list}
        else:
            tmp_dic_ = {"frame_id": 0,
                        "timestamp": str(td),
                        "gaze":[]}
        
        out_dic_list= [tmp_dic_]

        #########################################
        ## yjang inserted for writing results 
        self.jsonString = json.dumps(out_dic_list, indent=4)
        # jsonFile = open(out_info_file_path, "w")
        # jsonFile.write(self.jsonString)
        # # print(self.jsonString)
        # # loaded_dictionary = json.loads(self.jsonString)
        # jsonFile.close()
        #########################################

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

            if self.args.calib_file is not None:
                _dist_coefficients, _camera_matrix = self.load_camera_calibration(self.args.calib_file)
            else:
                im_width, im_height = self.img.shape[1], self.img.shape[0]
                tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
                _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                    [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

            self.estimate_gaze("None", self.img, _dist_coefficients, _camera_matrix)

    def start(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                cv2.imshow('Gaze', self.img)
                key = cv2.waitKey(1)
            if self.head_pose_image is not None:
                cv2.imshow('head',  self.head_pose_image)
                key = cv2.waitKey(1)
            if self.s_gaze_img is not None:
                cv2.imshow('gaze',  self.s_gaze_img)
                key = cv2.waitKey(1)
               
            
            if self.jsonString is not None:
                self.pub.publish(self.jsonString)
            
            self.loop_rate.sleep()


import sys

sys.argv[0]="tas_gaze.py"
sys.argv[1:]=["--gaze_backend",
"tensorflow"] # --gaze_backend has been only tested with tensorflow (not pytorch)


def main():
    args = parse_args()

    rospy.init_node('image_listener2', anonymous=True)

    CGaze_YJ = RT_GENE_GAZE_YJ(args)
    CGaze_YJ.start()




    # image_path_list = []
    # if os.path.isfile(args.im_path):
    #     image_path_list.append(os.path.split(args.im_path)[1])
    #     args.im_path = os.path.split(args.im_path)[0]
    # elif os.path.isdir(args.im_path):
    #     for image_file_name in sorted(os.listdir(args.im_path)):
    #         if image_file_name.lower().endswith('.jpg') or image_file_name.lower().endswith('.png') or image_file_name.lower().endswith('.jpeg'):
    #             if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
    #                 image_path_list.append(image_file_name)
    # else:
    #     tqdm.write('Provide either a path to an image or a path to a directory containing images')
    #     sys.exit(1)

    # if not os.path.isdir(args.output_path):
    #     os.makedirs(args.output_path)


if __name__ == '__main__':
    main ()