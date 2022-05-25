import os
import argparse
import sys

# sys.argv[0]="4_Switch_on_off_realsense_launch.py"
# sys.argv[1:]=["--realsense", "T"]

def parse_args():
    parser = argparse.ArgumentParser(description='Message Passing Framework')
    parser.add_argument('--realsense', type=str, default='F', help='realsense camera launch activation (T/F)')

    args = parser.parse_args()
    return args

# <include file="$(find realsense2_camera)/launch/rs_camera.launch"> <arg name="align_depth" value="true"/> <arg name="initial_reset" value="true"/> </include>

def switch_launch_modules_on_off(file_name, comparison_txt, switch):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if(comparison_txt in line):
                if(switch.upper() == 'T' or switch.upper() == 'TRUE'):
                    new_string = "    <include file=\"$(find realsense2_camera)/launch/rs_camera.launch\"> <arg name=\"align_depth\" value=\"true\"/> <arg name=\"initial_reset\" value=\"true\"/> </include>\n"
                else:
                    new_string = "    <!-- <include file=\"$(find realsense2_camera)/launch/rs_camera.launch\"> <arg name=\"align_depth\" value=\"true\"/> <arg name=\"initial_reset\" value=\"true\"/> </include>  -->\n"
                f.write(new_string)
            else:
                f.write(line)

                          
def main():
    
    args = parse_args()

    ## switching realsense_camera launch activation on and off    
    switch_launch_modules_on_off('./tas_perception/launch/main.copy.launch', "realsense2_camera", args.realsense[:1])

if __name__ == '__main__':
    main()
