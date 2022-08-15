##################33
####### To deal with the predicting data, just modify the data paths and save paths

import os
import numpy as np

from SimpleITK import ReadImage, GetArrayFromImage


datapath = 'D:/Leeds MS/cs/originaldatatrain/'  ###250 CASES original data
savepath = 'D:/Leeds MS/cs/traindata/plz/'  ###The save path of data

import imageio
import imageio

imgs = []
mha_path_list = []
label_path_list = []

T_list = os.listdir(datapath)


label_num = 0
raw_num = 0
for T_name in T_list:
    print(T_name)
    T_dir = datapath + T_name
    image_name = datapath + T_name + '/' + 'imaging.nii.gz'           ###Original CT Images
    mask_name = datapath + T_name + '/' + 'aggregated_OR_seg.nii.gz'  ###Mask CT Images
    rawdata = ReadImage(image_name)
    rawdata = GetArrayFromImage(rawdata)
    ####将超出部分与不足部分设置
    ### ###The kidneys have a high water content and are often examined with a window width of 200Hu-300Hu
    # and a window position of 25Hu-35Hu.
    rawdata[rawdata <= 25] = 25
    rawdata[rawdata >= 280] = 280
    rawdata = (rawdata - np.min(rawdata)) / (np.max(rawdata) - np.min(rawdata))
    rawdata = rawdata * 255

    ############预测使用
    ###
    # savedir1 = savepath+T_name
    # if not os.path.exists(savedir1):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(savedir1)

    # for depth in range(rawdata.shape[2]):
    #   rawslice = rawdata[:, :, depth]
    #   np.save('E:/chuliyuce/raw/rawnpy'+T_name+'/'+'%09d.npy' % raw_num, rawslice)
    #   raw_num = raw_num + 1


    for depth in range(rawdata.shape[2]):
      rawslice = rawdata[:, :, depth]
      # np.save('D:/AI/result/raw/'+'%06d.npy' % raw_num, rawslice)  #不加文件包
      np.save('D:/Leeds MS/cs/traindata/plz/rawnpy'+'/'+'%09d.npy' % raw_num, rawslice) ###rawnpy path
      raw_num = raw_num + 1

    ###The kidneys have a high water content and are often examined with a window width of 200Hu-300Hu
    # and a window position of 25Hu-35Hu.
    # savedir2 = savepath+T_name
    # if not os.path.exists(savedir2):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(savedir2)


    maskdata = ReadImage(mask_name)
    maskdata = GetArrayFromImage(maskdata)

    for depth1 in range(maskdata.shape[2]):
      masklice = maskdata[:, :, depth1]
      np.save('D:/Leeds MS/cs/traindata/plz/masknpy'+'/'+'%09d.npy' % label_num, masklice) ###masknpy path
      label_num = label_num + 1