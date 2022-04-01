# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/hand_object_detector')

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.net_utils import vis_detections_filtered_objects_PIL_saving  # yjang added for saving for integrated output
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

import time
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import String
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

import json
from datetime import timedelta

###########################################
# yjang added for intel realsen camera
import sys
sys.argv[0]="demo.py"
# sys.argv[1:]=["--cuda", "--webcam_num", "0", "--checkpoint", "132028", "--test_video_in", "None"]
sys.argv[1:]=["--cuda", "--webcam_num", "0", 
              "--realsense_camera", "False",
              "--cfg", "cfgs/res101.yml",
              "--net", "res101", ##'vgg16, res50, res101, res152'              
              "--runtime_img_in", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input.png",
              "--runtime_img_status", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input_status.txt",
              "--runtime_hand_states_json_out", "/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/hand_states_out.json"]
# sys.argv[1:]=["--cuda", "--webcam_num", "0", 
#               "--realsense_camera", "True"]
###########################################

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default="images_det")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=89999, type=int, required=False)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  parser.add_argument('--realsense_camera', default="False", type=str)  
  parser.add_argument('--test_video_in', default="/media/DataDisk/yj_bitbucket/dev/mmpose/vis_results/robot_cooking_videos/short_sample_clip.mp4", type=str)
  parser.add_argument('--hand_states_json_out', default="/media/DataDisk/yj_bitbucket/dev/hand_object_detector/info_results/hand_states_out.json", type=str)

  parser.add_argument('--runtime_img_in', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input.png", type=str)
  parser.add_argument('--runtime_hand_states_json_out', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/hand_states_out.json", type=str)
  parser.add_argument('--runtime_img_status', default="/media/DataDisk/yj_bitbucket/dev/TAS_integration/runtime_shared/input_status.txt", type=str)

  args = parser.parse_args()
  return args

class HandObjectDetector_YJ:
  def __init__(self, args):
    # params
    # self.lr = cfg.TRAIN.LEARNING_RATE
    # self.momentum = cfg.TRAIN.MOMENTUM
    # self.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    self.img = None
    self.jsonString = None

    self.loop_rate = rospy.Rate(30)  # ROS Rate at 5Hz

    self.down_sizing_factor = 0.5#0.5


    # Instantiate CvBridge
    self.bridge = CvBridge()

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    self.class_agnostic = args.class_agnostic
    self.cuda = args.cuda

    cfg.USE_GPU_NMS = self.cuda
    np.random.seed(cfg.RNG_SEED)
    

    self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

    # initilize the network here.
    if args.net == 'vgg16':
      self.fasterRCNN = vgg16(self.pascal_classes, pretrained=False, class_agnostic=self.class_agnostic)
    elif args.net == 'res101':
      self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
    elif args.net == 'res50':
      self.fasterRCNN = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=self.class_agnostic)
    elif args.net == 'res152':
      self.fasterRCNN = resnet(self.pascal_classes, 152, pretrained=False, class_agnostic=self.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    self.fasterRCNN.create_architecture()

    
    # load model
    # model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    # if not os.path.exists(model_dir):
    #   raise Exception('There is no input directory for loading network from ' + model_dir)
    # load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    load_name ="/ros-ws/src/tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth" ## faster_rcnn_1_8_132028.pth
    
    print("load checkpoint %s" % (load_name))
    if self.cuda > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    self.fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    self.im_data = torch.FloatTensor(1)
    self.im_info = torch.FloatTensor(1)
    self.num_boxes = torch.LongTensor(1)
    self.gt_boxes = torch.FloatTensor(1)
    self.box_info = torch.FloatTensor(1) 

    # ship to cuda
    if self.cuda > 0:
      self.im_data = self.im_data.cuda()
      self.im_info = self.im_info.cuda()
      self.num_boxes = self.num_boxes.cuda()
      self.gt_boxes = self.gt_boxes.cuda()
      self.box_info = self.box_info.cuda()

    with torch.no_grad():
      if self.cuda > 0:
        cfg.CUDA = True

      if self.cuda > 0:
        self.fasterRCNN.cuda()

      self.fasterRCNN.eval()

      # start = time.time()
      self.max_per_image = 100
      self.thresh_hand = args.thresh_hand 
      self.thresh_obj = args.thresh_obj
      self.vis = args.vis
      
      self.cfg = cfg ## yjang added to use cfg in the class

      image_topic = "/camera/color/image_raw"
      rospy.Subscriber(image_topic, Image, self.image_callback)

      self.pub = rospy.Publisher('hand_state_json_out', String, queue_size=10)
      
  def _get_image_blob(self, im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= self.cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in self.cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      # Prevent the biggest axis from being more than MAX_SIZE
      if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
        im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)
      im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
              interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

  def image_callback(self, msg):
    # print("Received an image!")
    try:
      # Convert your ROS Image message to OpenCV2
      color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
      cv2_img = cv2.resize(color_img, (int(color_img.shape[1]*self.down_sizing_factor), int(color_img.shape[0]*self.down_sizing_factor)))
    except CvBridgeError:
      print("e")
    else:
      img = cv2_img

      with torch.no_grad():
        try:
          blobs, im_scales = self._get_image_blob(img)
        except:
          print("break")

        # assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
          self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
          self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
          self.gt_boxes.resize_(1, 1, 5).zero_()
          self.num_boxes.resize_(1).zero_()
          self.box_info.resize_(1, 1, 5).zero_() 

        # pdb.set_trace()
        # det_tic = time.time()

        # print(self.im_data.is_cuda, self.im_info.is_cuda, self.gt_boxes.is_cuda, self.num_boxes.is_cuda, self.box_info.is_cuda)
        with torch.no_grad():
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label, loss_list = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info) 

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if self.cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if self.class_agnostic:
              if self.cuda > 0:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                          + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                          + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
            else:
              if self.cuda > 0:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                          + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                          + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        # det_toc = time.time()
        # detect_time = det_toc - det_tic
        # misc_tic = time.time()
        if self.vis:
          im2show = np.copy(img)
        obj_dets, hand_dets = None, None
        for j in xrange(1, len(self.pascal_classes)):
          # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
          if self.pascal_classes[j] == 'hand':
            inds = torch.nonzero(scores[:,j]>self.thresh_hand).view(-1)
          elif self.pascal_classes[j] == 'targetobject':
            inds = torch.nonzero(scores[:,j]>self.thresh_obj).view(-1)

          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if self.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], self.cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if self.pascal_classes[j] == 'targetobject':
              obj_dets = cls_dets.cpu().numpy()
            if self.pascal_classes[j] == 'hand':
              hand_dets = cls_dets.cpu().numpy()
              
        if self.vis:
          # visualization
          # im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, self.thresh_hand, self.thresh_obj)

          ##############################################################
          # yjang added for later visualisation at integrated version visualisation
          im2show, objests, hands = vis_detections_filtered_objects_PIL_saving(im2show, obj_dets, hand_dets, self.thresh_hand, self.thresh_obj)
          # import time
          t = time.localtime()
          td = time.strftime("%H:%M:%S.%f", t)
          tmp_dic_ = {"frame_id": 0,
                        "timestamp": td,
                        # "hand_states":{"hands":hands, "objects":objests}
                        "hands":hands, "objects":objests
                      }
          # out_dic_list.append(tmp_dic_)
          out_dic_list= [tmp_dic_]

          ##############################################################

        # misc_toc = time.time()
        # nms_time = misc_toc - misc_tic

        open_cv_image = np.array(im2show) 
        # im2showRGB = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        self.img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Hand_state", im2showRGB)

        # jsonString = json.dumps(out_dic_list, cls=MyEncoder, sort_keys=False, indent=2)
        self.jsonString = json.dumps(out_dic_list, indent=4)
        # jsonFile = open(out_info_file_path, "w")
        # jsonFile.write(jsonString)
        # jsonFile.close()      
        #########################################
  def start(self):
    while not rospy.is_shutdown():
      if self.img is not None:
          cv2.namedWindow('Hand_state')        # Create a named window
          cv2.moveWindow('Hand_state', 1500,600)  # Move it to (40,30)
          # self.pub_img.publish(self.br.cv2_to_imgmsg(self.image, encoding="rgb8"))

          # cv2.namedWindow('Body', cv2.WINDOW_NORMAL)

          cv2.imshow('Hand_state', self.img)
          # plt.figure(1); plt.clf()
          # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
          # plt.imshow(self.img)
          # plt.pause(0.03)
          
          key = cv2.waitKey(1)
      
      if self.jsonString is not None:
          # self.pub_coord.publish(self.obj_3d_coord)
          self.pub.publish(self.jsonString)
      
      self.loop_rate.sleep()

if __name__ == '__main__':
  import os
  curr_wd = os.getcwd()
  # Change the current working directory
  os.chdir('/hand_object_detector')

  args = parse_args()
  
  rospy.init_node('image_listener3', anonymous=True)

  C100HOD_YJ = HandObjectDetector_YJ(args)

  C100HOD_YJ.start()

  os.chdir(curr_wd)