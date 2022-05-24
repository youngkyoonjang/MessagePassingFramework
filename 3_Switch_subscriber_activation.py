import os
import argparse
import sys

# sys.argv[0]="2_Switch_subscriber_activation.py"
# sys.argv[1:]=["--pose", "F",
#               "--object", "T",
#               "--hand", "F",
#               "--integration", "F",
#               "--MPF", "F"]

def parse_args():
    parser = argparse.ArgumentParser(description='Message Passing Framework')
    parser.add_argument('--pose', type=str, default='F', help='pose estimation result subscriber activation (T/F)')
    parser.add_argument('--object', type=str, default='T', help='object detection result subscriber activation (T/F)')
    parser.add_argument('--hand', type=str, default='F', help='hand-object state detection result subscriber activation (T/F)')
    parser.add_argument('--integration', type=str, default='F', help='pose-object-hand subscriber and integration activation (T/F)')
    parser.add_argument('--MPF', type=str, default='F', help='MPF-applied process based on integrated results activation (T/F)')

    args = parser.parse_args()
    return args



def switch_launch_modules_on_off(file_name, comparison_txt, switch):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if(comparison_txt in line):
                if(switch.upper() == 'T' or switch.upper() == 'TRUE'):
                    if("p_subscriber" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_p_subscriber.py\" name=\"tas_p_subscriber\" />\n"
                    elif("o_subscriber" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_o_subscriber.py\" name=\"tas_o_subscriber\" />\n"
                    elif("h_subscriber" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_h_subscriber.py\" name=\"tas_h_subscriber\" />\n"
                    elif("hpo_integrator" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_hpo_integrator.py\" name=\"tas_hpo_integrator\" />\n"
                    elif("MPF_applied" in comparison_txt):
                        new_string = "    <node pkg=\"tas_perception\" type=\"tas_MPF_applied.py\" name=\"tas_MPF_applied\" />\n"
                else:
                    if("p_subscriber" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_p_subscriber.py\" name=\"tas_p_subscriber\" />  -->\n"
                    elif("o_subscriber" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_o_subscriber.py\" name=\"tas_o_subscriber\" />  -->\n"
                    elif("h_subscriber" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_h_subscriber.py\" name=\"tas_h_subscriber\" />  -->\n"
                    elif("hpo_integrator" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_hpo_integrator.py\" name=\"tas_hpo_integrator\" />  -->\n"
                    elif("MPF_applied" in comparison_txt):
                        new_string = "    <!-- <node pkg=\"tas_perception\" type=\"tas_MPF_applied.py\" name=\"tas_MPF_applied\" />  -->\n"
                f.write(new_string)
            else:
                f.write(line)

                          
def main():
    
    args = parse_args()

    ## switching module activation on and off    
    if(args.integration[:1] == 'T'): ## when pose estimation is activated, then it needs to activate object too!
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_p_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_o_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_h_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_hpo_integrator", 'T')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_MPF_applied", 'F')
    elif(args.MPF[:1] == 'T'): ## when pose estimation is activated, then it needs to activate object too!
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_p_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_o_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_h_subscriber", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_hpo_integrator", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_MPF_applied", 'T')
    else:
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_p_subscriber", args.pose[:1])
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_o_subscriber", args.object[:1])
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_h_subscriber", args.hand[:1])
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_hpo_integrator", 'F')
        switch_launch_modules_on_off('./tas_perception/launch/main.launch', "tas_MPF_applied", 'F')
    
if __name__ == '__main__':
    main()
