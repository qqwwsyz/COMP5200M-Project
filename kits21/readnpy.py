import os

import numpy as np
import cv2
from PIL import Image

### To generate the PNG files of train and predict, only thing to to is to change the paths.

datapath = 'D:/Leeds MS/cs/predictdata/plz/rawnpy/'  ###Read the path of npy files
savepath = 'D:/Leeds MS/cs/predictdata/plz/rawpng/'  ###Save path



imgs = []
mha_path_list = []
label_path_list = []

T_list = os.listdir(datapath)


label_num = 0
raw_num = 0
for T_name in T_list:
    # print(T_name)
    # savedir1 = savepath + T_name
    # if not os.path.exists(savedir1):  # Determine if a folder exists, if not then create as a folder
    #     os.makedirs(savedir1)

        T_dir = datapath + T_name
    # img_name = os.listdir(T_dir )
    # for i in img_name:
        all_name = datapath + T_name
        a = np.load(all_name)

        # a=a/60
        out=savepath+'/'+T_name.split('/')[-1].split('.')[0] +'.png'          ###Generate PNG files
        cv2.imwrite(out,a)