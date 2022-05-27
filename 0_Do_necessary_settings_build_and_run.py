import os
import sys

my_python_path = sys.executable

os.system(my_python_path + ' -m pip install tqdm')
os.system(my_python_path + ' 1_Download_files.py')
os.system(my_python_path + ' 2_Switch_module_activation_and_assign_gpus.py --pose F:0 --object T:0 --hand F:0 --gaze F:0') ##Acvitate:T/F, gpu_id:0/1
os.system(my_python_path + ' 3_Switch_subscriber_activation.py --pose F --object T --hand F --integration F --MPF F') ##Acvitate:T/F
os.system(my_python_path + ' 4_Switch_on_off_realsense_launch.py --realsense T') ##Acvitate:T/F
os.system(my_python_path + ' 5_Replace_perception_folder_path_in_Makefile.py')
os.system('make build && make run')

