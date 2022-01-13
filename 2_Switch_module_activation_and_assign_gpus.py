import os
import argparse
import sys

# sys.argv[0]="2_Switch_module_activation_and_assign_gpus.py"
# sys.argv[1:]=["--pose", "F:1",
#               "--object", "T:0",
#               "--hand", "F:0"]

def parse_args():
    parser = argparse.ArgumentParser(description='Message Passing Framework')
    parser.add_argument('--pose', type=str, default='F:1', help='pose estimation activation (T/F): gpu id (0/1)')
    parser.add_argument('--object', type=str, default='T:0', help='object detection activation (T/F): gpu id (0/1)')
    parser.add_argument('--hand', type=str, default='F:0', help='hand-object state detection activation (T/F): gpu id (0/1)')
    parser.add_argument('--gaze', type=str, default='F:0', help='gaze estimation activation (T/F): gpu id (0/1)')

    args = parser.parse_args()
    return args



def switch_launch_modules_on_off(file_name, comparison_txt, switch):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if(comparison_txt in line):
                if(switch.upper() == 'T' or switch.upper() == 'TRUE'):
                    if("pose" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_mmpose.py\" name=\"tas_pose\" />\n"
                    elif("object" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_mmdetection.py\" name=\"tas_object\" />\n"
                    elif("hand" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_hand.py\" name=\"tas_hand\" />\n"
                    elif("gaze" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_gaze.py\" name=\"tas_gaze\" />\n"
                else:
                    if("pose" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_mmpose.py\" name=\"tas_pose\" />  -->\n"
                    elif("object" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_mmdetection.py\" name=\"tas_object\" />  -->\n"
                    elif("hand" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_hand.py\" name=\"tas_hand\" />  -->\n"
                    elif("gaze" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_gaze.py\" name=\"tas_gaze\" />  -->\n"
                f.write(new_string)
            else:
                f.write(line)

def replace_gpu_asssignment_line(file_name, gpu_id):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if('cuda:' in line):
                if(gpu_id == 0):
                    new_string = line.replace("cuda:1", "cuda:0")
                elif(gpu_id == 1):
                    new_string = line.replace("cuda:0", "cuda:1")
                f.write(new_string)
            elif('gpu:' in line):
                if(gpu_id == 0):
                    new_string = line.replace("gpu:1", "gpu:0")
                elif(gpu_id == 1):
                    new_string = line.replace("gpu:0", "gpu:1")
                f.write(new_string)
            else:
                f.write(line)
                          
def main():
    
    args = parse_args()

    ## switching module activation on and off
    switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_pose", args.pose[:1])
    switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_object", args.object[:1])
    switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_hand", args.hand[:1])
    switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_gaze", args.gaze[:1])
    if(args.pose[:1] == 'T'): ## when pose estimation is activated, then it needs to activate object too!
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_object", 'T')
        
    ## assigning gpus for individual modules
    replace_gpu_asssignment_line('./tas_perception/scripts/tas_mmpose.py', int(args.pose[-1:]))
    replace_gpu_asssignment_line('./tas_perception/scripts/tas_mmdetection.py', int(args.object[-1:]))
    ## replace_gpu_asssignment_line('./tas_perception/scripts/tas_hand.py', int(args.hand[-1:])) ## I could not figure out how to assign for hand model yet
    replace_gpu_asssignment_line('./tas_perception/scripts/tas_gaze.py', int(args.gaze[-1:]))
    
    
    
if __name__ == '__main__':
    main()

