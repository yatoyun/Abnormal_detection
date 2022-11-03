from __future__ import absolute_import, print_function
import os
import utils
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
import scipy.io as sio
from options.testing_options import TestOptions
import utils
import time
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from utils import img_convert
from models import AutoEncoderCov3D, AutoEncoderCov3DMem, VariationalAutoEncoderCov3DMem

from piqa import SSIM

###
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

###
batch_size_in = opt.BatchSize  # 1
chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # frame number of the input images in a video clip
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

######
model_setting = utils.get_model_setting(opt)

# data path
data_root = opt.DataRoot + opt.Dataset + '/'
data_frame_dir = data_root + 'Test/'
data_idx_dir = data_root + 'Test_idx/'

# model path
model_root = opt.ModelRoot
if(opt.ModelFilePath):
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, 'model_' + model_setting + '/')


# test result path
te_res_root = opt.OutRoot
te_res_path = te_res_root + '/' + 'res_' + model_setting
utils.mkdir(te_res_path)

#loss
if opt.Loss == "ssim":
    tr_recon_loss_func = SSIM().to(device)
else:
    tr_recon_loss_func = nn.MSELoss(reduction='none')

# loading trained model
if (opt.ModelName == 'AE'):
    model = AutoEncoderCov3D(chnum_in_)
elif(opt.ModelName == 'MemAE'):
    model = AutoEncoderCov3DMem(
        chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
elif(opt.ModelName == 'MemVAE'):
    model = VariationalAutoEncoderCov3DMem(
        chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')
####
if opt.Test:
    load_state = torch.load(os.path.join(model_path, "test.pt"))
    model.load_state_dict(load_state)
else:
    if len(os.listdir(model_path)) != 0:
        print(model_path)
        PATH = os.listdir(model_path)[0]
        checkpoint = torch.load(os.path.join(model_path, PATH))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("There is no checkpoint of model")

##
model.to(device)
model.eval()

##
if(chnum_in_ == 1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_ == 3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# ##
video_list = utils.get_subdir_list(data_idx_dir)
video_num = len(video_list)

#threshold
threshold = 2*16*255

##
with torch.no_grad():
    for ite_vid in range(video_num):
        video_name = video_list[ite_vid]
        # idx path of the current sub dir
        video_idx_path = os.path.join(data_idx_dir, video_name)
        # frame path of the current sub dir
        video_frame_path = os.path.join(data_frame_dir, video_name)
        # info for current video
        idx_name_list = [name for name in os.listdir(video_idx_path)
                         if os.path.isfile(os.path.join(video_idx_path, name))]
        idx_name_list.sort()
        # load data (frame clips) for single video
        video_dataset = data.VideoDatasetOneDir(
            video_idx_path, video_frame_path, transform=frame_trans)
        video_data_loader = DataLoader(video_dataset,
                                       batch_size=batch_size_in,
                                       shuffle=False
                                       )
        # testing results on a single video sequence
        print('[vidx %02d/%d] [vname %s]' % (ite_vid+1, video_num, video_name))
        recon_error_list = []
        """fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存形式
        save = cv2.VideoWriter(os.path.join(
            te_res_path, video_name + '.avi'), fourcc, 10.0, (256, 256), 0)  # 動画を保存する形を作成"""
        #
        for batch_idx, (item, frames) in enumerate(video_data_loader):
            idx_name = idx_name_list[item[0]]
            idx_data = sio.loadmat(os.path.join(video_idx_path, idx_name))
            v_name = idx_data['v_name'][0]  # video name
            # frame index list for a video clip
            frame_idx = idx_data['idx'][0, :]
            ######
            frames = frames.to(device)
            ##
            if (opt.ModelName == 'AE'):
                recon_frames = model(frames)
                # calculate reconstruction error (MSE)
                recon_np = utils.vframes2imgs(unorm_trans(
                    recon_frames.data), step=1, batch_idx=0)
                input_np = utils.vframes2imgs(
                    unorm_trans(frames.data), step=1, batch_idx=0)
                r = utils.crop_image(recon_np, img_crop_size) - \
                    utils.crop_image(input_np, img_crop_size)
                # recon_error = np.mean(sum(r**2)**0.5)
                recon_error = np.mean(r ** 2)  # **0.5
                recon_error_list += [recon_error]
            elif (opt.ModelName == 'MemAE'):
                recon_res, _ = model(frames)
                # recon_frames = recon_res['output']
                #print(recon_frames.size(), frames.size())
                r = recon_frames - frames
                # image
                #frames_recon_vis = utils.vframes2imgs(unorm_trans(recon_frames.data), step=5, batch_idx=0)
                #frames_recon_vis = np.concatenate(frames_recon_vis, axis=-1)
                #frames_recon_vis = frames_recon_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                utils.mkdir(os.path.join(te_res_path, video_name))

                #recon_frames = recon_frames.cpu().numpy()
                #print("size : ", recon_frames.size())
                #for k, (recon_frame, frame) in enumerate(np.array(recon_frames.squeeze()), np.array(frames.squeeze())):
                mseimgs = []
                for i in range(recon_frames.shape[2]):
                    #mseimg = (loss_func_mse(recon_frames[0,:,i], frames[0,:,i])[0].cpu().detach().numpy())

                    #mseimg = mseimg[:,:,np.newaxis]
                    #mseimg = (mseimg - np.min(mseimg)) / (np.max(mseimg)-np.min(mseimg))
                    img = frames[0,:,i].cpu().numpy().squeeze()
                    output = recon_frames[0,:,i].cpu().numpy().squeeze()
                    mseimg = np.abs(output*255-img*255)
                    mseimg = mseimg.astype(dtype=np.uint8)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    #plot
                    plt.imshow(img, cmap=plt.cm.gray) #interpolation='nearest'
                    H = signal.convolve2d(mseimg.squeeze(), np.ones((6,6)), mode='same')
                    x, y = np.where(H > threshold)
                    plt.scatter(y,x,color='red',s=0.04)
                    plt.axis('off')

                    plt.savefig(os.path.join(os.path.join(te_res_path,video_name), 'MSE_{:04d}.jpg').format(batch_idx))
                    plt.close()
                    mseimgs.append(mseimg)
                """mseimgs = np.array(mseimgs).squeeze()
                recon_frames = recon_frames.squeeze().cpu().numpy()
                frames = frames.squeeze().cpu().numpy()
                #color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
                #print("mse shape : ",mseimgs.shape)
                #print("frame shape : ", recon_frames.shape)
                for (org_img, f_diff) in zip(frames, mseimgs):
                    #color_map = plt.get_cmap("viridis")
                    heatmap = cv2.applyColorMap(f_diff, cv2.COLORMAP_JET)
                    heatmap = cv2.blur(heatmap,(9,9))
                    mask = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold((mask), 200, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    #heatmap = np.uint8(color_map(f_diff * 255))
                    heatmap2 = cv2.applyColorMap(f_diff*255, cv2.COLORMAP_HOT)
                    org_img = cv2.cvtColor(org_img*255,cv2.COLOR_GRAY2RGB)
                    img1_bg = cv2.bitwise_and(org_img,org_img,mask = mask_inv).astype(dtype=np.uint8)
                    img2_fg = cv2.bitwise_and(heatmap2,heatmap2,mask = mask)
                    #print("img1_bg shape : ",img1_bg.shape)
                    #print("img2_fg shape : ", img2_fg.shape)
                    dst = cv2.add(img1_bg,img2_fg)
                    transposed = org_img#.transpose(1, 2, 0)[:, :, [2, 1, 0]]
                    resized = cv2.resize(
                        heatmap, (transposed.shape[0], transposed.shape[1])
                    )
                    transposed = cv2.cvtColor(transposed, cv2.COLOR_GRAY2RGB)
                    blended = cv2.addWeighted(
                        transposed, 1.0, resized, 0.01, 2.2, dtype=cv2.CV_32F
                    )
                    blended_normed = (
                        255 * (blended - blended.min()) /
                        (blended.max() - blended.min())
                    )
                    blended_out = np.array(blended_normed, np.int64)"""
                    #cv2.imwrite(os.path.join(os.path.join(te_res_path,video_name), 'MSE_{:04d}.jpg').format(batch_idx), blended_out)
                    # save.write(np.array(img_convert.toimage(img)))
                ###
                r = utils.crop_image(r, img_crop_size)
                sp_error_map = torch.sum(r**2, dim=1)**0.5
                s = sp_error_map.size()
                sp_error_vec = sp_error_map.view(s[0], -1)
                recon_error = torch.mean(sp_error_vec, dim=-1)
                recon_error_list += recon_error.cpu().tolist()
            elif(opt.ModelName == 'MemVAE'):
                recon_res, _ = model(frames)
                recon_frames = recon_res
                #print(recon_frames.size(), frames.size())
                r = recon_frames - frames
                # image
                #frames_recon_vis = utils.vframes2imgs(unorm_trans(recon_frames.data), step=5, batch_idx=0)
                #frames_recon_vis = np.concatenate(frames_recon_vis, axis=-1)
                #frames_recon_vis = frames_recon_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                utils.mkdir(os.path.join(te_res_path, video_name))

                #recon_frames = recon_frames.cpu().numpy()
                #print("size : ", recon_frames.size())
                #for k, (recon_frame, frame) in enumerate(np.array(recon_frames.squeeze()), np.array(frames.squeeze())):
                mseimgs = []
                for i in range(recon_frames.shape[2]):
                    #mseimg = (loss_func_mse(recon_frames[0,:,i], frames[0,:,i])[0].cpu().detach().numpy())

                    #mseimg = mseimg[:,:,np.newaxis]
                    #mseimg = (mseimg - np.min(mseimg)) / (np.max(mseimg)-np.min(mseimg))
                    img = frames[0,:,i].cpu().numpy().squeeze()
                    output = recon_frames[0,:,i].cpu().numpy().squeeze()
                    mseimg = np.abs(output*255-img*255)
                    mseimg = mseimg.astype(dtype=np.uint8)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    #plot
                    plt.imshow(img, cmap=plt.cm.gray) #interpolation='nearest'
                    H = signal.convolve2d(mseimg.squeeze(), np.ones((6,6)), mode='same')
                    x, y = np.where(H > threshold)
                    plt.scatter(y,x,color='red',s=0.04)
                    plt.axis('off')

                    plt.savefig(os.path.join(os.path.join(te_res_path,video_name), 'MSE_{:04d}.jpg').format(batch_idx))
                    plt.close()
                    mseimgs.append(mseimg)
                r = utils.crop_image(r, img_crop_size)
                sp_error_map = torch.sum(r**2, dim=1)**0.5
                s = sp_error_map.size()
                sp_error_vec = sp_error_map.view(s[0], -1)
                recon_error = torch.mean(sp_error_vec, dim=-1)
                recon_error_list += recon_error.cpu().tolist()
            ######
            # elif (opt.ModelName == 'MemAE'):
            #     recon_res = model(frames)
            #     recon_frames = recon_res['output']
            #     recon_np = utils.btv2btf(unorm_trans(recon_frames.data))
            #     input_np = utils.btv2btf(unorm_trans(frames.data))
            #     r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
            #     sp_error_map = np.sum(r**2, axis=1)**0.5
            #     tmp_s = sp_error_map.shape
            #     sp_error_vec = np.reshape(sp_error_map, (tmp_s[0], -1))
            #     recon_error = np.mean(sp_error_vec, axis=-1)
            #     recon_error_list += recon_error.tolist()
            #######
            else:
                recon_error = -1
                print('Wrong ModelName.')
        np.save(os.path.join(te_res_path, video_name + '.npy'), recon_error_list)
        # save.release()


# evaluation
utils.eval_video(data_root, te_res_path, is_show=True)
