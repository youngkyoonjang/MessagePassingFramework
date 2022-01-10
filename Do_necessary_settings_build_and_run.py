import os

os.system('python Download_files.py')
os.system('python Switch_module_activation_and_assign_gpus.py --pose T:1 --object T:0 --hand F:0') ##Acvitate:T/F, gpu_id:0/1
os.system('python Replace_perception_folder_path_in_Makefile.py')
os.system('make build && make run')

