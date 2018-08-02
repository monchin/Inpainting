from model import Context_Encoder, Adversarial_Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import utils
import numpy as np
import pandas as pd



# No overlap
def joint_loss_no_overlap(input_img, hole_size=64, lambda_recon=0.999):
    mask_recon = np.ones(shape=(hole_size, hole_size, 3))
    mask_all = np.pad(mask_recon, 128-hole_size, "constant")
    mask_context = 1-mask_all
    input_img_context = input_img*mask_context
    input_img_hole = input_img[(128-hole_size)//2:(128+hole_size)//2, 
                               (128-hole_size)//2:(128+hole_size)//2]

    context = Context_Encoder()
    recon_hole = context(input_img_context)

    adversarial = Adversarial_Discriminator()
    adversarial_real = adversarial(input_img_hole)
    adversarial_recon = adversarial(recon_hole)

    recon_loss_fun = nn.MSELoss()
    recon_loss = recon_loss_fun(recon_hole, input_img_hole)

    adver_loss_fun = nn.BCEWithLogitsLoss()
    adver_loss = adver_loss_fun(adversarial_recon, adversarial_real)

    joint_loss = lambda_recon*recon_loss + (1-lambda_recon)*adver_loss

    return joint_loss


def train():
    if checkPath is not None:
        state = torch.load(checkPath)
        model.load_state_dict(state['state_dict'])

    mask_recon = np.ones(shape=(hole_size, hole_size))
    mask_all = np.pad(mask_recon, (128-hole_size)//2, "constant")
    mask_all = np.expand_dims(mask_all, 2)
    mask_all = np.stack([mask_all*3], 2)
    mask_all = np.expand_dims(mask_all, 0)
    mask_all = np.vstack([mask_all*batch_size])
    mask_context = 1-mask_all

    context = Context_Encoder()
    adversarial = Adversarial_Discriminator()

    optimizer_context = torch.optim.Adam(params=context.parameters(), 
                                         lr=lr_context)
    optimizer_adver = torch.optim.Adam(params=adversarial.parameters(),
                                       lr=lr_adver)

    recon_loss_fun = nn.MSELoss()
    adver_loss_fun = nn.BCEWithLogitsLoss()

    label_real = torch.ones(batch_size)
    label_fake = torch.zeros(batch_size)

    if device is not None:
        recon_loss_fun = recon_loss_fun.cuda(device)
        adver_loss_fun = adver_loss_fun.cuda(device)

    for each_epoch in range(n_epoch):
        # input_img = ?

        input_img_context = input_img*mask_context
        input_img_context = torch.from_numpy(input_img_context)
        input_img_hole = input_img[:, (128-hole_size)//2:(128+hole_size)//2, 
                               (128-hole_size)//2:(128+hole_size)//2, :]
        input_img_hole = torch.from_numpy(input_img_hole)

        if device is not None:
            input_img_context = input_img_context.cuda(device)
            input_img_hole = input_img_hole.cuda(device)
        
        recon_hole = context(input_img_context)

        adversarial_real = adversarial(input_img_hole)
        adversarial_recon = adversarial(recon_hole)

        recon_loss = recon_loss_fun(recon_hole, input_img_hole)

