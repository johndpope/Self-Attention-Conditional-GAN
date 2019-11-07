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

parser = argparse.ArgumentParser()
parser.add_argument('--n_cl', type=int, default=0)
parser.add_argument('--G', type=str, default=None)
parser.add_argument('--att', type=bool, default=False)
parser.add_argument('--trun', type=float, default=None)

args = parser.parse_args()

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

def get_noise(b_size,z_dim=128):
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

def get_netG(n_classes = n_cl ,ch = 68,att = args.att):
    netG = sn_res.Generator64(n_classes = n_classes ,ch = ch,att = att).to(device)
    #netG = nn.DataParallel(netG, list(range(ngpu)))

    if args.G == None:
    # original saved file with DataParallel
        state_dict = torch.load('sn_res_64_cgan_49_Gen.pth')
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

def gen_imgs():
    noise = get_noise(64)
    l = []
    fig = plt.figure(figsize=(10,10))
    if n_cl>0:
        y1,y1_ohe = sample_pseudo_labels(n_cl,64)
    else:
        y1,y1_ohe = None,None

    with torch.no_grad():
      fake = netG(noise,y1_ohe).detach().cpu()
      l.append(vutils.make_grid(fake, padding=2, normalize=True))
    plt.imshow(np.transpose(l[-1],(1,2,0)))
    fig.savefig('plot.png')

def gen_cond_imgs():
    z = get_noise(100)
    y =  [i for i in range (2,12) for _ in range(10)]
    #y = 13*np.ones(100,dtype =  np.int8)
    y_ohe = np.eye(n_cl)[y]
    y_ohe = torch.from_numpy(y_ohe)
    y_ohe = y_ohe.float().to(device)

    #labels = sample_pseudo_labels(n_cl, n_cl)
    with torch.no_grad():
        sample_images = netG(z, y_ohe).detach().cpu()
    grid = vutils.make_grid(sample_images, nrow=10,normalize=True).permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(grid)
    fig.savefig('cond_plot.png')
    
gen_imgs()
if n_cl>0:
    gen_cond_imgs()

