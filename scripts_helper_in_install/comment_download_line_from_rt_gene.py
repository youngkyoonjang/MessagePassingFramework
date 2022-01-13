def comment_download_line_from_rt_gene(file_name, comparison_txt):
    with open(file_name, "r") as f:
        contents = f.readlines()

    with open(file_name, "w") as f:
        for line in contents:
            if(comparison_txt in line):
                new_string = line[:1] + '#' + line[1:]
                f.write(new_string)
            else:
                f.write(line)

comment_download_line_from_rt_gene('./extract_landmarks_method_base.py', "download_external_landmark_models()")