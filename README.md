# MessagePassingFramework
Message Passing Framework for Vision Prediction Stability in HRI

# Requirements 
RGB-D camera is required. Specifically we tested using Inteal RealSense D435.

# Running Instruction: 
1. Download github repository: git clone https://github.com/youngkyoonjang/MessagePassingFramework.git
2. Change parameters depending on what you want to run in '0_Do_necessary_settings_build_and_run.py':
If you want to run pose estimation using the first GPU: --pose T:0 --object F:0 --hand F:0 --gaze F:0
If you want to run both object detection on the first GPU and hand-object state estimation on the second GPU: --pose F:0 --object T:0 --hand T:1 --gaze F:0
Keep in mind that each model requires GPU memory to load their models, so you are only able to run modules depending on your GPU memory capacity.
3. Do everything else: python 0_Do_necessary_settings_build_and_run.py

# Notes: 
* It is still work-in-progress repository. Please keep it confidential. I am sharing this only within PRL Lab only.
* It has some additional features, such as message passsing between modules using ROS publisher/subscriber. And I am willing to update it as requested by PRL Lab members. So feel free to share your request to use this repository. If it is already there, I will put more explanation how to access the data. Otherwise, I can add the functionality and share it for you. 
* If you found other modules useful, ask me to consider. Then, I will update it throughout my stay at PRL.