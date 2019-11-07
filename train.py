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
from collections import OrderedDict
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
import sys
from models import sn_cnn512, sn_cnn1024,sn_res
from datasets import Dogs_labels
from datasets.Dogs_labels import DogsDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--loss', type=str, default='HINGE')
parser.add_argument('--model', type=str, default='sn_cnn_64_512')
parser.add_argument('--disc_iters', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--data', type=str, default='dogs')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--cgan', type=bool, default=False)
parser.add_argument('--att', type=bool, default=False)
parser.add_argument('--G_ch', type=int, default=64)
parser.add_argument('--D_ch', type=int, default=64)
parser.add_argument('--lr_G', type=float, default=7e-5)
parser.add_argument('--lr_D', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--leak', type=float, default=0)
parser.add_argument('--loaded_G', type=str, default=None)
parser.add_argument('--loaded_D', type=str, default=None)

args = parser.parse_args()

# Device
ngpu = args.ngpu
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

#Seeds
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#parameters
batch_size = args.batch_size
disc_iters = args.disc_iters
loss_fun = args.loss
epochs = args.epochs
cgan =  args.cgan
loaded_G = args.loaded_G
loaded_D = args.loaded_D
z_dim = 128
channels = 3

#hyperparameres
lr_D = args.lr_D
lr_G = args.lr_G
beta1 = args.beta1
beta2 = args.beta2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


print("Data laoding... ")

#dataset
if args.data == 'dogs':
    train_data = Dogs_labels.DogsDataset()
    dataloader = torch.utils.data.DataLoader(train_data,
                           shuffle=True, batch_size=batch_size,
                           num_workers=12,pin_memory=True)

elif  args.data == 'cifar':
    train_data = dset.CIFAR10(root='./data', train=True, download=True,
                               transform=stransforms.Compose([
                                   # transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

print("Finished data loading")
# conditional GAN
if cgan:
    n_cl = len(train_data.classes)
else:
    n_cl = 0


def sample_pseudo_labels(num_classes, batch_size):
    y =  torch.randint(low=0, high=num_classes, size=(batch_size,)).to(device,non_blocking=True)
    y_ohe = torch.eye(num_classes)[y].to(device,non_blocking=True)
    y = y.type(torch.long)
    return y,y_ohe


def sample_from_gen(b_size, nz, num_classes, gen):
    z = torch.randn(b_size, nz).to( device=device,non_blocking=True)
    if cgan:
        y,y_ohe = sample_pseudo_labels(num_classes, b_size)
    else:
        y,y_ohe = None,None

    fake = gen(z, y_ohe)

    return fake, y

#model
if args.model == 'sn_cnn_64_512':
  netG = sn_cnn512.Generator(z_dim,channels).to(device)
  netD = sn_cnn512.Discriminator(channels).to(device)
  netG.apply(weights_init)
  netD.apply(weights_init)
elif args.model == 'sn_cnn_64_1024':
  netG = sn_cnn1024.Generator(z_dim,channels).to(device)
  netD = sn_cnn1024.Discriminator(channels).to(device)
  netG.apply(weights_init)
  netD.apply(weights_init)
elif args.model == 'sn_res_64':
  netG = sn_res.Generator64(z_dim,channels,n_classes = n_cl,ch = args.G_ch,leak = args.leak,att = args.att).to(device,non_blocking=True)
  netD = sn_res.Discriminator64(channels,n_classes = n_cl,ch = args.D_ch,leak = args.leak,att = args.att).to(device,non_blocking=True)
elif args.model == 'sn_res_32':
    netG = sn_res.Generator32(z_dim, channels,n_classes = n_cl,ch = args.G_ch,leak = args.leak,att = args.att).to(device)
    netD = sn_res.Discriminator32(channels,n_classes = n_cl,ch = args.D_ch,leak = args.leak,att = args.att).to(device)


if loaded_G is not None:
    state_dict_G = torch.load(loaded_G)
    state_dict_D = torch.load(loaded_D)
    new_state_dict_G = OrderedDict()
    for k, v in state_dict_G.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict_G[k] = v
    netG.load_state_dict(new_state_dict_G)
    netG.eval()
    new_state_dict_D = OrderedDict()
    for k, v in state_dict_D.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict_D[k] = v
    netD.load_state_dict(new_state_dict_D)
    netD.eval()
    

# Parallel GPU if ngpu > 1
if (device.type == 'cuda') and (ngpu > 1):
  netG = nn.DataParallel(netG, list(range(ngpu)))
  netD = nn.DataParallel(netD, list(range(ngpu)))

# Print the model
print(netG)
print(netD)
print(sum(p.numel() for p in netG.parameters()))
print(sum(p.numel() for p in netD.parameters()))


# Testing architecture
noise = torch.randn(1, z_dim, device=device)
if n_cl>0:
    y1,y1_ohe = sample_pseudo_labels(n_cl,1)
else:
    y1,y1_ohe = None,None   
fake = netG(noise,y1_ohe)
d_out = netD(fake,y1)
print(fake.size(),d_out.size())

#settings for adv loss
if loss_fun == 'adv':
    dis_criterion = nn.BCEWithLogitsLoss().to(device,non_blocking=True)
    #labels
    label_t = 0.9
    label_f = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))

# use decaying learning rate
#schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
#schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

MILESTONES = [30,60,90,120] #None
SCHEDULER_GAMMA = 0.4
if MILESTONES is not None:
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=MILESTONES, 
                                                gamma=SCHEDULER_GAMMA, last_epoch=-1)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=MILESTONES, 
                                                gamma=SCHEDULER_GAMMA, last_epoch=-1)
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


TIME_LIMIT = 32400
start_time = time.time()
def elapsed_time(start_time):
    return time.time() - start_time

def train(num_epochs=1, disc_iters=1):
    global img_list, G_losses, D_losses, iters
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        if elapsed_time(start_time) > TIME_LIMIT:
            print('Time limit reached')
            break
        D_running_loss = 0
        G_running_loss = 0
        # For each mini-batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            real_x = data[0].to(device,non_blocking=True)
            if n_cl > 0:
                    real_y = data[1].long().to(device,non_blocking=True)
            else :
                real_y = None
            b_size = real_x.size(0)
            
            # Update D network
            for _ in range(disc_iters): 
                netD.zero_grad()
                
                real_logit = netD(real_x, real_y)
                if loss_fun == 'HINGE':
                    D_loss_real = torch.mean(F.relu(1.0 - real_logit))
                if loss_fun == 'adv':
                    adv_labels = torch.full((b_size, 1), label_t, device=device)
                    D_loss_real = dis_criterion(real_logit,adv_labels)
                D_loss_real.backward()
                    
                fake_x, fake_y = sample_from_gen(b_size,z_dim, n_cl, netG)
                fake_logit = netD(fake_x.detach(),fake_y)
                if loss_fun == 'HINGE':
                    D_loss_fake = torch.mean(F.relu(1.0 + fake_logit))
                if loss_fun == 'adv':
                    adv_labels.fill_(label_f)
                    D_loss_fake = dis_criterion(fake_logit,adv_labels)                             
                D_loss_fake.backward()
              
                optimizerD.step()
                D_running_loss += (D_loss_fake.item() + D_loss_real.item())/len(dataloader)
                
           # Update G
            netG.zero_grad()
            
            fake_x, fake_y = sample_from_gen(b_size, z_dim, n_cl, netG)
            fake_logit = netD(fake_x,fake_y)

            if loss_fun == 'HINGE':
                _G_loss = -torch.mean(fake_logit)
            if loss_fun == 'adv':
                adv_labels.fill_(label_t)  
                _G_loss = dis_criterion(fake_logit, adv_labels)                          
            _G_loss.backward()
            optimizerG.step()
            G_running_loss += _G_loss.item()/len(dataloader)
            
            iters += 1
        
        if MILESTONES is not None:
            schedulerD.step()
            schedulerG.step()
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f, elapsed_time = %.4f min'
              % (epoch, num_epochs,
                 D_running_loss, G_running_loss,elapsed_time(start_time)/60))

        # Save Losses for plotting later
        G_losses.append(G_running_loss)
        D_losses.append(D_running_loss)


# saving and showing results
filename = args.model + '_' + str(args.epochs)
if cgan:
    filename = args.model + '_cgan_' + str(args.epochs)

train(epochs,disc_iters)

torch.save(netG.state_dict(),filename +"_Gen.pth")
torch.save(netD.state_dict(),filename +"_Dis.pth")

fig1 = plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig1.savefig(filename + 'losses.png')

