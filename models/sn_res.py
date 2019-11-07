import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F

def conv3x3(ch_in,ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1,bias = True)

def conv1x1(ch_in,ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0,bias = True)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
            
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        
        
class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel, affine=False)  # no learning parameters
        self.embed = nn.Linear(n_condition, in_channel * 2) # part of w and part of bias

        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        embed = self.embed(label.float())
        gamma, beta = embed.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = (1+gamma) * out + beta
        return out
    
class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.phi      = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.g        = nn.utils.spectral_norm(conv1x1(channels, channels//2)).apply(init_weight)
        self.o        = nn.utils.spectral_norm(conv1x1(channels//2, channels)).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
    
    
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_channels=None, upsample=False,n_classes = 0, leak = 0):
        super(ResBlockGenerator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample
        
        self.conv1 = SpectralNorm(conv3x3(in_channels,hidden_channels)).apply(init_weight)
        self.conv2 = SpectralNorm(conv3x3(hidden_channels,out_channels)).apply(init_weight)
        self.conv3 = SpectralNorm(conv1x1(in_channels,out_channels)).apply(init_weight)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.n_cl = n_classes
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:
            self.bn1 = ConditionalNorm(in_channels,n_classes)
            self.bn2 = ConditionalNorm(hidden_channels,n_classes)

        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = self.upsampling(x)
                x = self.conv3(x)
            else:
                x = self.conv3(x)
            return x
        else:
            return x

    def forward(self, x,y=None):
        if y is not None:
            out = self.activation(self.bn1(x,y))
        else:
            out = self.activation(self.bn1(x))

        if self.upsample:
             out = self.upsampling(out)

        out = self.conv1(out)
        if y is not None:
            out = self.activation(self.bn2(out,y))
        else:
            out = self.activation(self.bn2(out))

        out = self.conv2(out)
        out_res = self.shortcut(x)
        return out + out_res

class Generator32(nn.Module):
    def __init__(self,z_dim =128,channels=3,ch = 32,n_classes = 0,leak =0):
        super(Generator32, self).__init__()

        self.ch = ch
        self.dense = SpectralNorm(nn.Linear(z_dim, 4 * 4 * 8*ch)).apply(init_weight)
        self.final = SpectralNorm(conv3x3(ch,channels)).apply(init_weight)

        self.block1 = ResBlockGenerator(8*ch, 4*ch,upsample=True,n_classes = n_classes,leak = leak)
        self.block2 = ResBlockGenerator(4*ch, 2*ch,upsample=True,n_classes = n_classes,leak = leak)
        self.block3 = ResBlockGenerator(2*ch, ch,upsample=True,n_classes = n_classes,leak = leak)

        self.bn = nn.BatchNorm2d(ch)
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()   

    def forward(self, z,y=None):
        h = self.dense(z).view(-1, 8*self.ch, 4, 4)
        h = self.block1(h,y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)
    
class Generator64(nn.Module):
    def __init__(self,z_dim =128,channels=3,ch = 64,n_classes = 0,leak = 0,att = False):
        super(Generator64, self).__init__()

        self.ch = ch
        self.n_classes = n_classes
        self.att = att
        self.dense = SpectralNorm(nn.Linear(z_dim, 4 * 4 * ch*8)).apply(init_weight)
        self.final = SpectralNorm(conv3x3(ch,channels)).apply(init_weight)

        self.block1 = ResBlockGenerator(ch*8, ch*8,upsample=True,n_classes = n_classes,leak = leak)
        self.block2 = ResBlockGenerator(ch*8, ch*4,upsample=True,n_classes = n_classes,leak = leak)
        self.block3 = ResBlockGenerator(ch*4, ch*2,upsample=True,n_classes = n_classes,leak = leak)
        if att:
            self.attention = Attention(ch*2)
        self.block4 = ResBlockGenerator(ch*2, ch,upsample=True,n_classes = n_classes,leak = leak)

        self.bn = nn.BatchNorm2d(ch)
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()          
    def forward(self, z,y=None):
        h = self.dense(z).view(-1,self.ch*8, 4, 4)
        h = self.block1(h,y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        if self.att:
            h = self.attention(h)
        h = self.block4(h,y)
        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,hidden_channels=None,leak = 0):
        super(ResBlockDiscriminator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv1 =  SpectralNorm(conv3x3(in_channels, hidden_channels)).apply(init_weight)
        self.conv2 = SpectralNorm(conv3x3(hidden_channels, out_channels)).apply(init_weight)
        self.conv3 = SpectralNorm(conv1x1(in_channels, out_channels)).apply(init_weight)

        self.learnable_sc = (in_channels != out_channels) or downsample
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        self.downsampling = nn.AvgPool2d(2)
        self.downsample = downsample

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsampling(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv3(x)
            if self.downsample:
                return self.downsampling(x)
            else:
                return x
        else:
            return x

    def forward (self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels,leak =0):
        super(OptimizedBlock, self).__init__()
        self.conv1 = SpectralNorm(conv3x3(in_channels, out_channels)).apply(init_weight)
        self.conv2 = SpectralNorm(conv3x3(out_channels, out_channels)).apply(init_weight)
        self.conv3 = SpectralNorm(conv1x1(in_channels, out_channels)).apply(init_weight)
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            nn.AvgPool2d(2)  # stride = 2 ( default = kernel size)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.conv3
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator32(nn.Module):
    def __init__(self, channels=3,ch = 32,n_classes=0,leak =0):
        super(Discriminator32, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
        
        self.ch = ch
        self.model = nn.Sequential(
            OptimizedBlock(channels, ch,leak = leak),
            ResBlockDiscriminator(ch, ch*2, downsample=True,leak = leak),
            ResBlockDiscriminator(ch*2, ch*4,downsample=True,leak = leak),
            ResBlockDiscriminator(ch*4, ch*8,leak = leak),
            self.activation,
        )
        self.fc =  SpectralNorm(nn.Linear(ch*8, 1)).apply(init_weight)
        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes,ch*8).apply(init_weight)

    def forward(self, x,y=None):
        h = torch.sum(self.model(x), dim=(2, 3))
        h = h.view(-1, self.ch*8)
        output = self.fc(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

class Discriminator64(nn.Module):
    def __init__(self, channels=3,ch = 32,n_classes=0,leak =0,att = False):
        super(Discriminator64, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
            
        self.ch = ch
        self.att = att
        self.block1=OptimizedBlock(channels, ch,leak = leak)
        if att:
            self.attention = Attention(ch)
        self.block2=ResBlockDiscriminator(ch, ch*2, downsample=True,leak = leak)
        self.block3=ResBlockDiscriminator(ch*2 , ch*4,downsample=True,leak = leak)
        self.block4=ResBlockDiscriminator(ch*4, ch*8,downsample=True,leak = leak)
        self.block5=ResBlockDiscriminator(ch* 8, ch*16,leak = leak)
            
        self.fc =  SpectralNorm(nn.Linear(self.ch*16, 1)).apply(init_weight)
        if n_classes > 0:
                self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)

    def forward(self, x,y=None):
        h = self.block1(x)
        if self.att:
            h = self.attention(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h,dim = (2,3))
        
        h = h.view(-1, self.ch*16)
        
        output = self.fc(h)
        if y is not None:
            output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)
        return output

# Testing architecture

'''
D = Discriminator64()
G = Generator64()

noise = torch.randn(1, 128)
fake = G(noise)
d_out = D(fake)
print(fake.size(),d_out.size()) '''
