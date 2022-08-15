import SimpleITK
import numpy as np
import imageio
def save_mha(Array, filepath,fromtype='Array',Mode='seg'):
    if fromtype == 'Array':
        if not Mode=='seg':
            Array = Array
        if Mode=='seg':
            Array=Array.astype(np.ubyte)
        src_array_min = Array.min()
        itk_img = SimpleITK.GetImageFromArray(Array)
        Spacing = [0.3516, 0.3516, 4.39956]
        itk_img.SetSpacing(Spacing)
        #scaledpacing=[2.0,2.0,2.0]
        # itk_img=inverse_scale_size2(itk_img,scaledpacing, np.double(src_array_min))
    else:
        itk_img = Array
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(filepath)
     # ?_t1ce?????
    size = itk_img.GetSize()
    origin = itk_img.GetOrigin()
    spacing = itk_img.GetSpacing()
    direction = itk_img.GetDirection()
    writer.Execute(itk_img)


import os
datapath = 'D:/Leeds MS/cs/rawpng/'
doc_list = os.listdir(datapath)

for doc_name in doc_list:
    print(doc_name)
    doc_dir = datapath+'/'+doc_name
    pic_list = os.listdir(doc_dir)
    i = 0
    save_mhd = np.zeros((512,512,835))


    for pic_name in pic_list:
        pic_dir = doc_dir + '/' + pic_name
        data = imageio.imread(pic_dir)
        save_mhd[:,:,i] = data
        save_mha(save_mhd,'D:/Leeds MS/cs/rawmhd/'+doc_name+'.mhd')
        i = i+1





