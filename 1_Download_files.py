import os
from pathlib import Path
import requests

from rt_gene.download_tools import download_external_landmark_models, download_gaze_tensorflow_models


def replace_the_first_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        f.write(line + '\n')
        cnt=0
        for line in contents:
            if cnt==0:
                cnt+=1
            else:
                f.write(line)
                
def exchange_strings(file_name, txt_to_be_replaced, with_txt_given):
    """ Insert given string as a new line at the beginning of a file """
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if '../../../../_base_' in line:
                new_line = line.replace(txt_to_be_replaced, with_txt_given)
                f.write(new_line)
            else:
                f.write(line)

Path('./tas_perception/models').mkdir(parents=True, exist_ok=True)


models_directory_path = "./tas_perception/models"

# Download MMDetection model and configuration files: YOLACT
if not os.path.isfile(models_directory_path+'/yolact_r50_1x8_coco.py'):
    os.system('wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolact/yolact_r50_1x8_coco.py -P ' + models_directory_path)
if not os.path.isfile(models_directory_path+'/yolact_r101_1x8_coco.py'):
    os.system('wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolact/yolact_r101_1x8_coco.py -P ' + models_directory_path)
if not os.path.isfile(models_directory_path+'/yolact_r101_1x8_coco_20200908-4cbe9101.pth'):
    os.system('wget https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth  -P ' + models_directory_path)

# Download MMPose model and configuration files: HRNet (Topdown Heatmap + Hrnet on Coco -- pose_hrnet_w48 256x192)
if not os.path.isfile(models_directory_path+'/hrnet_w48_coco_256x192.py'):
    os.system('wget https://raw.githubusercontent.com/open-mmlab/mmpose/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py -P ' + models_directory_path)
    # replace_the_first_line(models_directory_path+'/hrnet_w48_coco_256x192.py', "_base_ = ['../_base_/datasets/coco.py']")
    exchange_strings(models_directory_path+'/hrnet_w48_coco_256x192.py', "../../../../_base_", "../_base_")

if not os.path.isfile(models_directory_path+'/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'):
    os.system('wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth  -P ' + models_directory_path)


# Download Hand-Object State Estimation model: FasterRCNN

# -------------------------------------------------- Downloading googld drive file

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
# --------------------------------------------------

Path('./tas_perception/hand_object/models').mkdir(parents=True, exist_ok=True)

hand_model_directory_path = "./tas_perception/hand_object/models"
if not os.path.isfile(hand_model_directory_path+'/faster_rcnn_1_8_89999.pth'):
    file_id = '166IM6CXA32f9L6V7-EMd9m8gin6TFpim'
    destination = hand_model_directory_path+'/faster_rcnn_1_8_89999.pth'
    print('./tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth is downloading please wait until it is completed!')
    download_file_from_google_drive(file_id, destination)
    print('./tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth download complete!!')

# Download rt_gene model /rt_gene_/model_nets folder
Path('/rt_gene/model_nets').mkdir(parents=True, exist_ok=True)
Path('./rt_gene/model_nets/SFD').mkdir(parents=True, exist_ok=True)
Path('./rt_gene/model_nets/ThreeDDFA').mkdir(parents=True, exist_ok=True)

download_external_landmark_models()
download_gaze_tensorflow_models()