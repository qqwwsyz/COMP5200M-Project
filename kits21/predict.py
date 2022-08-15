# import argparse
# import logging
# import os
#
# import PIL
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms
# import cv2
# from utils.data_loading import BasicDataset
# from unet import UNet
# from utils.utils import plot_img_and_mask
#
# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=0.5,
#                 out_threshold=0.5):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
#
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)
#
#
#     with torch.no_grad():
#         output = net(img)
#         if net.n_classes > 1:
#             probs = F.softmax(output, dim=1)[0]
#             probs = probs.cpu().numpy()
#             probs = np.argmax(probs,0)
#             probs = cv2.resize(probs,(512,512),interpolation = cv2.INTER_NEAREST)
#             return probs
#
#
# name = 'MR466142'
# def get_args():
#
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='./checkpoints_Bone_da/checkpoint_epoch4.pth', metavar='FILE',  #1210 17
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--output', '-o', default='E:/chuliyuce/segresult/'+name
#                         ,metavar='INPUT', nargs='+', help='Filenames of output images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#
#     return parser.parse_args()
#
# def get_output_filenames(args):
#     def _generate_name(fn):
#         split = os.path.splitext(fn)
#         return f'{split[0]}_OUT{split[1]}'
#
#     return args.output or list(map(_generate_name, args.input))
#
# def mask_to_image(mask: np.ndarray):
#     if mask.ndim == 2:
#         return Image.fromarray((mask * 30).astype(np.uint8))
#     elif mask.ndim == 3:
#         a = np.argmax(mask, axis=0)
#         return (np.argmax(mask, axis=0) * 30).astype(np.uint8)
#         # return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
#
# if __name__ == '__main__':
#     args = get_args()
#     in_files = 'E:/chuliyuce/raw'
#     out_files = get_output_filenames(args)
#     net = UNet(n_channels=1, n_classes=4, bilinear=args.bilinear)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')
#
#     net.to(device=device)
#     net.load_state_dict(torch.load(args.model, map_location=device))
#
#     logging.info('Model loaded!')
#
#
#     in_files_doclist = os.listdir(in_files)
#
#     in_files_list = []
#     for docname in in_files_doclist:
#         doc_name_dir = in_files+'/'+docname
#
#
#         raw_name_list = os.listdir(doc_name_dir)
#         in_files_list = []
#         #
#         for raw_name in raw_name_list:
#
#             dir = doc_name_dir+'/' + raw_name
#             in_files_list.append(dir)
#
#
#         for i, filename in enumerate(in_files_list):
#             logging.info(f'\nPredicting image {filename} ...')
#          #   img = Image.open(filename)
#             img = np.load(filename)
#
#
#
#             mask = predict_img(net=net,
#                                full_img=img,
#                                scale_factor=args.scale,
#                                out_threshold=args.mask_threshold,
#                                device=device)
#             if not os.path.exists('E:/chuliyuce/segresult/'+docname):
#                 os.makedirs('E:/chuliyuce/segresult/'+docname)
#             if not args.no_save:
#             #     # out_filename ='./result/seg'+docname+'/'+filename.split('/')[-1]
#                 out_filename = 'E:/chuliyuce/segresult/' + docname+'/'+filename.split('/')[-1].split('.')[0]+'.png'
#                 mask = mask*60
#                 out_filename = out_filename
#                 cv2.imwrite(out_filename,mask)
#
#
#
#                 #np.save('D:/AI/result/seg/' + docname + '/' + 'filename.split('/')[-1]', mask)
#
#             #     logging.info(f'Mask saved to {out_filename}')
#             # if args.viz:
#             #     logging.info(f'Visualizing results for image {filename}, close to continue...')
#             #     plot_img_and_mask(img, mask)




import argparse
import logging
import os

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from utils.data_loading4Bone import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)


    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
            probs = probs.cpu().numpy()
            probs = np.argmax(probs,0)
            probs = cv2.resize(probs,(512,512),interpolation = cv2.INTER_NEAREST)
            return probs


name = 'MR466142'
def get_args():

    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints_Bone_da/checkpoint_epoch4.pth', metavar='FILE',  #You can change the models here
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output', '-o', default='D:/Leeds MS/cs/predictdata/plz/prepng/'+name
                        ,metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 30).astype(np.uint8))
    elif mask.ndim == 3:
        a = np.argmax(mask, axis=0)
        return (np.argmax(mask, axis=0) * 30).astype(np.uint8)
        # return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

if __name__ == '__main__':
    args = get_args()
    # in_files = 'D:/Leeds MS/cs/predictdata/raw/rawpredict'
    # in_files = 'D:/Leeds MS/cs/predictdata/raw/rawpng'
    in_files = 'D:/Leeds MS/cs/predictdata/plz/rawpng'     ###To read the PNG files
    out_files = get_output_filenames(args)
    net = UNet(n_channels=1, n_classes=4, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')


    in_files_doclist = os.listdir(in_files)

    # in_files_list = []
    # for docname in in_files_doclist:
    #     doc_name_dir = in_files+'/'+docname

        #
        # raw_name_list = os.listdir(doc_name_dir)
        # in_files_list = []
        # #
        # for raw_name in raw_name_list:
        #
        #     dir = doc_name_dir+'/' + raw_name
        #     in_files_list.append(dir)


        # for i, filename in enumerate(in_files_list):
    for filename in in_files_doclist:
            # logging.info(f'\nPredicting image {filename} ...')
            filenamee=in_files +'/'+ filename
            img = Image.open(filenamee)
           # img = np.load(filename)



            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            # if not os.path.exists('D:/Leeds MS/cs/segresult'+docname):
            #     os.makedirs('D:/Leeds MS/cs/segresult'+docname)
            # if not args.no_save:
            # #     # out_filename ='./result/seg'+docname+'/'+filename.split('/')[-1]
            # #     out_filename = 'D:/Leeds MS/cs/segresult' + docname+'/'+filename.split('/')[-1].split('.')[0]+'.png'
            # out_filename='D:/Leeds MS/cs/predictdata/raw/rawsegresult/'+filename.split('/')[-1]
            out_filename = 'D:/Leeds MS/cs/predictdata/plz/prepng_model4/' + filename.split('/')[-1] #Generate output of predicting PNGs
            mask = mask*60

            out_filename = out_filename
            cv2.imwrite(out_filename,mask)


                #np.save('D:/AI/result/seg/' + docname + '/' + 'filename.split('/')[-1]', mask)

            #     logging.info(f'Mask saved to {out_filename}')
            # if args.viz:
            #     logging.info(f'Visualizing results for image {filename}, close to continue...')
            #     plot_img_and_mask(img, mask)