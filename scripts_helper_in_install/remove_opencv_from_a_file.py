with open("./requirements.txt","r+") as f:
    new_f = f.readlines()
    f.seek(0)
    for line in new_f:
        if "opencv" not in line:
            f.write(line)
    f.truncate()