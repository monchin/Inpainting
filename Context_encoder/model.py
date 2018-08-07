import torch
import torch.nn as nn
import torch.nn.functional as F


class Context_Encoder(nn.Module):

    def __init__(self, conv_activate=nn.LeakyReLU, conv_trans_activate=nn.ReLU, 
                 BN=True):
        super(Context_Encoder, self).__init__()        
        self.encoder = nn.Sequential(
            Conv(3, 64, activate=conv_activate, BN=BN),
            Conv(64, 64, activate=conv_activate, BN=BN),
            Conv(64, 128, activate=conv_activate, BN=BN),
            Conv(128, 256, activate=conv_activate, BN=BN),
            Conv(256, 512, activate=conv_activate, BN=BN),
            Conv(512, 4000, stride=1, padding=0, activate=conv_activate, 
                    BN=BN)
        )
        self.decoder = nn.Sequential(
            Conv_Trans(4000, 512, stride=1, padding=0, 
                       activate=conv_trans_activate, BN=BN),
            Conv_Trans(512, 256, activate=conv_trans_activate, BN=BN),
            Conv_Trans(256, 128, activate=conv_trans_activate, BN=BN),
            Conv_Trans(128, 64, activate=conv_trans_activate, BN=BN),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        )

    def forward(self, inputs):
        x = self.encoder(inputs)
        y = self.decoder(x)
        
        return y



class Adversarial_Discriminator(nn.Module):

    def __init__(self, activate=nn.ReLU, BN=True):
        super(Adversarial_Discriminator, self).__init__()
        self.adversarial = nn.Sequential(
            Conv(3, 64, activate=activate, BN=BN),
            Conv(64, 128, activate=activate, BN=BN),
            Conv(128, 256, activate=activate, BN=BN),
            Conv(256, 512, activate=activate, BN=BN),
        )
        self.fully_connected = nn.Linear(4*4*512, 1)

    def forward(self, inputs):
        x = self.adversarial(inputs)
        x = x.view(-1, 4*4*512)        
        result = self.fully_connected(x)
        return result


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, activate, kernel_size=4, 
                 stride=2, padding=1, BN=True):
        super(Conv, self).__init__()
        if (BN==True):
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activate(inplace=True)
            )
        elif (BN==False):
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                          stride=stride, padding=padding),
                activate(inplace=True)
            )

    def forward(self, inputs):
        return self.conv(inputs)


class Conv_Trans(nn.Module):
    
    def __init__(self, in_channels, out_channels, activate, kernel_size=4, 
                 stride=2, padding=1, BN=True):
        super(Conv_Trans, self).__init__()
        if (BN==True):
            self.conv_trans = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activate(inplace=True)
            )
        elif (BN==False):
            self.conv_trans = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding),
                activate(inplace=True)
            )

    def forward(self, inputs):
        return self.conv_trans(inputs)
