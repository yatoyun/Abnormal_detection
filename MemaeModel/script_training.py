import os
import utils
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
import scipy.io as sio
from options.training_options import TrainOptions
import utils
import time
from models import AutoEncoderCov3D, AutoEncoderCov3DMem, VariationalAutoEncoderCov3DMem
from models import EntropyLossEncap, KLDLoss
from summary import summary
from piqa.ssim import SSIM

###
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

###
utils.seed(opt.Seed)
if(opt.IsDeter):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

######
model_setting = utils.get_model_setting(opt)
print('Setting: %s' % (model_setting))

############
batch_size_in = opt.BatchSize
learning_rate = opt.LR
max_epoch_num = opt.EpochNum

chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # num of frames in a video clip
mem_dim_in = opt.MemDim
entropy_loss_weight = opt.EntropyLossWeight
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

print('bs=%d, lr=%f, entrloss=%f, shr=%f, memdim=%d' % (batch_size_in,
      learning_rate, entropy_loss_weight, sparse_shrink_thres, mem_dim_in))
############
# data path
data_root = opt.DataRoot + opt.Dataset + '/'
tr_data_frame_dir = data_root + 'Train/'
tr_data_idx_dir = data_root + 'Train_idx/'

# model saving dir path
saving_root = opt.ModelRoot
saving_model_path = os.path.join(saving_root, 'model_' + model_setting + '/')
utils.mkdir(saving_model_path)

# tblog
if(opt.IsTbLog):
    log_path = os.path.join('./log/', 'log_'+model_setting + '/')
    utils.mkdir(log_path)
    tb_logger = utils.Logger(log_path)

##
if(chnum_in_ == 1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_ == 2):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
elif(chnum_in_ == 3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# data
video_dataset = data.VideoDataset(
    tr_data_idx_dir, tr_data_frame_dir, transform=frame_trans)
tr_data_loader = DataLoader(video_dataset,
                            batch_size=batch_size_in,
                            shuffle=True,
                            num_workers=opt.NumWorker
                            )


# model
if(opt.ModelName == 'MemAE'):
    model = AutoEncoderCov3DMem(
        chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
elif(opt.ModelName == 'MemVAE'):
    model = VariationalAutoEncoderCov3DMem(
        chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong model name.')
model.apply(utils.weights_init)

# optimizer
tr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# device
device = torch.device("cuda" if use_cuda else "cpu")

# reset
last_epoch = 0
global_ite_idx = 0
PATH = "model_" + opt.ModelName
if len(os.listdir(saving_model_path)) != 0 and opt.UseExModel:
    print(saving_model_path)
    PATH = os.listdir(saving_model_path)[0]
    checkpoint = torch.load(os.path.join(saving_model_path, PATH))
    model.load_state_dict(checkpoint['model_state_dict'])
    tr_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in tr_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_ite_idx = checkpoint['global_ite_idx']


#########
model.to(device)
if opt.Loss == "ssim":
    tr_recon_loss_func = SSIM(n_channels=1).to(device)
    from piqa.utils import set_debug
    set_debug(False)
else:
    tr_recon_loss_func = nn.MSELoss().to(device)
tr_entropy_loss_func = EntropyLossEncap().to(device)
latent_loss_weight = 0.25


# model summary
if opt.Summary:
    summary(model, (1, 16, 256, 256))

#loss

##
data_loader_len = len(tr_data_loader)
textlog_interval = opt.TextLogInterval
snap_save_interval = opt.SnapInterval
save_check_interval = opt.SaveCheckInterval
tb_img_log_interval = opt.TBImgLogInterval
for epoch_idx in range(last_epoch, max_epoch_num):
    for batch_idx, (item, frames) in enumerate(tr_data_loader):
        #frames = frames.squeeze().reshape(256, 256, 16, -1)
        frames = frames.to(device) #[14, 1, 16, 256, 256]
        if (opt.ModelName == 'MemAE'):
            recon_res = model(frames)
            recon_frames = recon_res['output']
            att_w = recon_res['att']
            if opt.Loss == "ssim":
                loss = 1 - tr_recon_loss_func(recon_frames, frames)
            else:
                loss = tr_recon_loss_func(recon_frames, frames)
            recon_loss_val = loss.item()
            entropy_loss = tr_entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss = loss + entropy_loss_weight * entropy_loss

            loss_val = loss.item()
            ##
            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()
            ##
        elif(opt.ModelName == "MemVAE"):
            #recon_res, latent_loss, att_w = model(frames)
            recon_frames, latent_loss = model(frames)
            # recon_frames = recon_res['output']
            # att_w = recon_res['att']
            # loss = tr_recon_loss_func(recon_frames, frames)
            # recon_loss_val = loss.item()
            # recon_loss_val = recon_loss_val #+ KLD
            # entropy_loss = tr_entropy_loss_func(att_w)
            # entropy_loss_val = entropy_loss.item()
            # loss = loss + entropy_loss_weight * entropy_loss
            if opt.Loss == "ssim":
                recon_loss_val = 1 - tr_recon_loss_func(recon_frames, frames)
            else:
                recon_loss_val = tr_recon_loss_func(recon_frames, frames)
            #latent
            try:
                latent_loss = latent_loss.mean()
            except:
                latent_loss =  0
            #entorpy
            # entropy_loss = tr_entropy_loss_func(att_w)
            # entropy_loss_val = entropy_loss.item()
            #total loss
            loss = recon_loss_val + latent_loss * latent_loss_weight #+ entropy_loss_weight * entropy_loss
            loss_val = loss.item()
            ##
            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()
        # TB log val
        if(opt.IsTbLog):
            tb_info = {
                'loss': loss_val,
                'recon_loss': recon_loss_val,
                #'entropy_loss': entropy_loss_val
            }
            for tag, value in tb_info.items():
                tb_logger.scalar_summary(tag, value, global_ite_idx)
            # TB log img
            if((global_ite_idx % tb_img_log_interval) == 0):
                frames_vis = utils.vframes2imgs(unorm_trans(frames.data), step=5, batch_idx=0)
                frames_vis = np.concatenate(frames_vis, axis=-1)
                frames_vis = frames_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                frames_recon_vis = utils.vframes2imgs(unorm_trans(recon_frames.data), step=5, batch_idx=0)
                frames_recon_vis = np.concatenate(frames_recon_vis, axis=-1)
                frames_recon_vis = frames_recon_vis[None, :, :] * np.ones(3, dtype=int)[:, None, None]
                tb_info = {
                    'x': frames_vis,
                    'x_rec': frames_recon_vis
                }
                for tag, imgs in tb_info.items():
                    tb_logger.image_summary(tag, imgs, global_ite_idx)
        ##
        if((batch_idx % textlog_interval) == 0):
            print('[%s, epoch %d/%d, bt %d/%d] loss=%f, rc_losss=%f, ent_loss=0'% (model_setting, epoch_idx,
                  max_epoch_num, batch_idx, data_loader_len, loss_val, recon_loss_val))

        if((global_ite_idx % snap_save_interval) == 0):
            utils.mkdir(saving_model_path)
            torch.save({'epoch': epoch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': tr_optimizer.state_dict(),
                        'loss': loss,
                        'global_ite_idx': global_ite_idx,
                        },
                       '%s/%s' % (saving_model_path,PATH))
        global_ite_idx += 1

    if((epoch_idx % save_check_interval) == 0):
        torch.save({'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': tr_optimizer.state_dict(),
                    'loss': loss,
                    'global_ite_idx': global_ite_idx,
                    },
                   '%s/%s' % (saving_model_path,PATH))

torch.save({'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': tr_optimizer.state_dict(),
            'loss': loss,
            'global_ite_idx': global_ite_idx,
            },
           '%s/%s' % (saving_model_path,PATH))
