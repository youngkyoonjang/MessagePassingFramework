import os

def replace_the_perception_folder_path_line(file_name, comparison_txt):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if(comparison_txt in line):
                # str_curwd = os.path.abspath(os.getcwd())
                # new_string = "\t\t-v "+str_curwd+"/tas_perception/"+":/ros-ws/src/tas_perception/ \\\n"
                new_string = "\t\t-v "+"tas_perception"+":/ros-ws/src/tas_perception/ \\\n"
                f.write(new_string)
            else:
                f.write(line)

replace_the_perception_folder_path_line('./Makefile', ":/ros-ws/src/tas_perception/")
