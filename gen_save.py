import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import time
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn.utils.spectral_norm as SpectralNorm
from models import *
from models import sn_cnn512, sn_cnn1024,sn_res
from collections import OrderedDict
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--n_cl', type=int, default=0)
parser.add_argument('--G', type=str, default=None)
parser.add_argument('--att', type=bool, default=False)
parser.add_argument('--trun', type=float, default=None)
parser.add_argument('--fname', type=str, default='out')

args = parser.parse_args()
out_path = args.fname
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
z_d = 128
channels = 3
ngpu = 2
n_cl = args.n_cl


def sample_pseudo_labels(num_classes, batch_size):
    y =  torch.randint(low=0, high=num_classes, size=(batch_size,)).to(device,non_blocking=True)
    y_ohe = torch.eye(num_classes)[y].to(device,non_blocking=True)
    y = y.type(torch.long)
    return y,y_ohe

def get_trun(b_size,z_dim=128):
    truncated = args.trun
    if truncated is not None:
        flag = True
        while flag:
            z = np.random.randn(100*b_size*z_dim)
            z = z[np.where(abs(z)<truncated)]
            if len(z)>=64*z_dim:
                flag=False
        gen_z = torch.from_numpy(z[:b_size*z_dim]).view(b_size,z_dim)
        gen_z = gen_z.float().to(device)
    else:
        gen_z = torch.randn(b_size, z_dim, device=device)
    return gen_z

def submission_generate_images(truncated=None):
    im_batch_size=50
    n_images=10000
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for i_batch in tqdm(range(0, n_images, im_batch_size)):
        if truncated is not None:
            gen_z = get_trun(im_batch_size)
        else:
            gen_z = torch.randn(im_batch_size, 128, device=device)
            
        if n_cl>0:
            y1,y1_ohe = sample_pseudo_labels(n_cl,im_batch_size)
        else:
            y1,y1_ohe = None,None
        
        gen_images = netG(gen_z,y1_ohe)
        gen_images = gen_images.to("cpu").clone().detach() #shape=(*,3,h,w), torch.Tensor
        #denormalize
        gen_images = gen_images*0.5 + 0.5
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :],
                       os.path.join(out_path, f'image_{i_batch+i_image:05d}.png'))
    #shutil.make_archive(f'images', 'zip', out_path)

    
def get_netG(n_classes = n_cl ,ch = 68,att = args.att):
    #netG = sn_res.Generator64(n_classes = n_classes ,ch = ch,att = att).to(device)
    #netG = nn.DataParallel(netG, list(range(ngpu)))
    netG = sn_cnn512.Generator(128,3).to(device)

    if args.G == None:
    # original saved file with DataParallel
        state_dict = torch.load('sn_res_64_cgan_27_Gen.pth')
    else:
        state_dict = torch.load(args.G)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict[k] = v

    netG.load_state_dict(new_state_dict)
    netG.eval()
    
    return netG

netG = get_netG(n_classes = n_cl ,ch = 68,att = args.att)

submission_generate_images(args.trun)