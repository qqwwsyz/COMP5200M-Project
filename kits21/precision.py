# import os
# import SimpleITK as sitk
# import numpy as np
# from PIL import Image
# from SimpleITK import GetArrayFromImage
#
#
#
# pre_list = os.listdir(predir)
#
# ave=[]
# for name in pre_list:
#   doc_name = predir + name
#   T_name=os.listdir(doc_name)
#   ave_1=[]
#   for pict in T_name:
#     preim = Image.open(doc_name+'/'+pict)
#     # rawdata = ReadImage(image_name)
#     preim = np.array(preim)
#     preim = preim / 60
#
#     maskim=Image.open(label_dir+name+'/'+pict)
#     maskim = np.array(maskim)
#
#     ###寻找相同的
#     label = [1,2,3]
#     dice = []
#     temp=[]
#     for i in range(len(label)):
#       intersection = np.sum((preim == label[i]) * (maskim == label[i]))
#       union = np.sum(preim == label[i]) + np.sum(maskim == label[i])
#
#
#       dice.append(2 * intersection / union)
#
#     dice=np.array(dice)
#     dice[np.isnan(dice)] = 1
#     temp=np.mean(dice)
#
#
#     ave_1.append(temp)
#   temp1 = np.mean(ave_1)
#   ave.append(temp1)
#   print(ave)
# AVE=np.mean(ave)
# print('ave is %f' %(AVE))




import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
from SimpleITK import GetArrayFromImage

predir = 'D:/Leeds MS/cs/predictdata/plz/prepng_model4/'  ###The path of Predicting PNG Files
label_dir ='D:/Leeds MS/cs/predictdata/plz/maskpng/'      ###The standard answers from the challenge holders


pre_list = os.listdir(predir)

ave=[]
ave_1=[]

for name in pre_list:
    doc_name = predir + name
  # T_name=os.listdir(doc_name)
  # for pict in T_name:
    preim = Image.open(doc_name)
    # rawdata = ReadImage(image_name)
    preim = np.array(preim)
    preim = preim / 60
    # preim_max= np.max(preim)
    # print(preim_max)

    maskim=Image.open(label_dir+'/'+name)
    maskim = np.array(maskim)
    # mask_max = np.max(maskim)
    # print(mask_max)

#     ###寻找相同的
#     label = [1,2,3]
#     dice = []
#     temp=[]
#     for i in range(len(label)):
#       intersection = np.sum((preim == label[i]) * (maskim == label[i]))
#       union = np.sum(preim == label[i]) + np.sum(maskim == label[i])
#       dice.append(2 * intersection / union)
#
#     dice=np.array(dice)
#     # print(dice)
#     dice[np.isnan(dice)] = 1
#     # print(temp)
#     temp = np.mean(dice)
#     ave_1.append(temp)
#     # print(ave_1)
#     # temp1 = np.mean(ave_1)
#    # ave.append(temp1)
#   # print(ave)
# AVE=np.mean(ave_1)
# print('ave is %f' %(AVE))



#     if np.max(maskim)>0:
#
#       ###寻找相同的
#       label = [1,2,3]
#       dice = []
#       temp=[]
#       for i in range(len(label)):
#         intersection = np.sum((preim == label[i]) * (maskim == label[i]))
#         union = np.sum(preim == label[i]) + np.sum(maskim == label[i])
#         dice.append(2 * intersection / union)
#
#       dice=np.array(dice)
#       dice[np.isnan(dice)] = 1
#       temp=np.mean(dice)
#       ave_1.append(temp)
#       # print(ave_1)
#     # temp1 = np.mean(ave_1)
#   # ave.append(temp1)
#   # print(ave)
# AVE=np.mean(ave_1)
# print('ave is %f' %(AVE))


##############Determining separate categories
    # Determine if the object has a target
    if 3 in maskim:

      ###寻找相同的
      label = [1,2,3]
      dice = []
      temp=[]
      for i in range(len(label)):
        intersection = np.sum((preim == label[i]) * (maskim == label[i]))
        union = np.sum(preim == label[i]) + np.sum(maskim == label[i])
        dice.append(2 * intersection / union)

      dice=np.array(dice)
      dice[np.isnan(dice)] = 1
      temp=np.mean(dice)

      ave_1.append(temp)
length=len(ave_1)
print(length)
    # temp1 = np.mean(ave_1)
  # ave.append(temp1)
  # print(ave)
AVE=np.mean(ave_1)
print('ave is %f' %(AVE)) #Generate the ave dice score

