import os
import numpy as np

savepath1 = 'D:/Leeds MS/cs/traindata/raw'  ###读取文件地址
savepath2 = 'D:/Leeds MS/cs/traindata/mask'  ###读取文件地址

filelist = os.listdir(savepath1)

for f in filelist:
    filepath1 =os.path.join(savepath1,f)
    img1= np.load(filepath1)
    if (img1.shape[0] == 512) and (img1.shape[1] == 512):
      np.save('D:/Leeds MS/cs/traindata1/raw/'+f.split('/')[-1].split('.')[0]+'.npy', img1)
    else:
      print(img1.shape)

    filepath2 = os.path.join(savepath2, f)
    img2 = np.load(filepath2)
    if (img2.shape[0] == 512) and (img2.shape[1] == 512):
      np.save('D:/Leeds MS/cs/traindata1/mask/'+ f.split('/')[-1].split('.')[0] + '.npy', img2)


