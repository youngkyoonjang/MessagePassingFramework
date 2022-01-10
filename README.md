# MessagePassingFramework
Message Passing Framework for Vision Prediction Stability in HRI

# Requirements 
Docker is required.
RGB-D camera is required. Specifically we tested using Inteal RealSense D435.


In the process of docker installation, this script will download:
YOLACT config file_dependency: https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolact/yolact_r50_1x8_coco.py
YOLACT config file: https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolact/yolact_r101_1x8_coco.py
YOLACT model file: https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth
that you can download from: https://github.com/open-mmlab/mmdetection/blob/master/configs/yolact/README.md
for object detection. 
You can also download other model config and model from the mmdetection website: https://github.com/open-mmlab/mmdetection


In order to run: python Do_necessary_settings_build_and_run.py
This will do following:
1. Download model files: python Download_files.py 
1. Download model files: python Replace_perception_folder_path_in_Makefile.py
2. Build Docker and run: make build && make run