from model import Context_Encoder, Adversarial_Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import numpy as np



def train(hole_size=64, batch_size=100, lr_context=0.001, lr_adver=0.0001,
          lambda_recon=0.999, n_epoch=500, device=3, check_path=None):
    
    context = Context_Encoder()
    adversarial = Adversarial_Discriminator()

    if check_path is not None:
        state = torch.load(check_path)
        context.load_state_dict(state['state_dict_c'])
        adversarial.load_state_dict(state['state_dict_a'])

    mask_recon = np.ones(shape=(hole_size, hole_size), dtype=np.float32)
    mask_all = np.pad(mask_recon, (128-hole_size)//2, "constant")
    mask_all = np.expand_dims(mask_all, 0)
    mask_all = np.concatenate([mask_all]*3, 0)
    print(mask_all.shape)
    mask_all = np.expand_dims(mask_all, 0)
    mask_all = np.vstack([mask_all]*batch_size)
    mask_context = 1-mask_all
    mask_context = torch.from_numpy(mask_context)
    print(mask_context.shape)


    optimizer_context = torch.optim.Adam(params=context.parameters(), 
                                         lr=lr_context)
    optimizer_adver = torch.optim.Adam(params=adversarial.parameters(),
                                       lr=lr_adver)

    recon_loss_fun = nn.MSELoss()
    adver_loss_fun = nn.BCEWithLogitsLoss()

    label_real = torch.ones([batch_size,1])
    label_fake = torch.zeros([batch_size,1])

    if device is not None:
        mask_context = mask_context.cuda(device)
        recon_loss_fun = recon_loss_fun.cuda(device)
        adver_loss_fun = adver_loss_fun.cuda(device)
        label_real = label_real.cuda(device)
        label_fake = label_fake.cuda(device)
        context = context.cuda(device)
        adversarial = adversarial.cuda(device)

    start = default_timer()
    for each_epoch in range(n_epoch):
        # # just for test
        # input_img = torch.randint(0, 255, size=(batch_size, 3, 128, 128),
        #                           dtype=torch.float32)
        if device is not None:
            input_img = input_img.cuda(device)

        input_img_context = input_img*mask_context
        input_img_hole = input_img[:, :, (128-hole_size)//2:(128+hole_size)//2, 
                                   (128-hole_size)//2:(128+hole_size)//2]

        optimizer_context.zero_grad()

        recon_hole = context(input_img_context)
        recon_loss = recon_loss_fun(recon_hole, input_img_hole)

        recon_loss_lambda = lambda_recon*recon_loss
        #recon_loss_lambda.backward()
        #optimizer_context.step()


        optimizer_adver.zero_grad()

        adversarial_real = adversarial(input_img_hole)
        adversarial_recon = adversarial(recon_hole)

        adver_real_loss = adver_loss_fun(adversarial_real, label_real)
        adver_recon_loss = adver_loss_fun(adversarial_recon.detach(), label_fake)
        adver_loss = adver_real_loss+adver_recon_loss

        adver_loss_lambda = (1-lambda_recon)*adver_loss
        #adver_loss_lambda.backward()

        total_loss = recon_loss_lambda+adver_loss_lambda
        total_loss.backward()
        
        optimizer_context.step()
        optimizer_adver.step()

        
        if each_epoch%10==9:
            total_loss = recon_loss_lambda+adver_loss_lambda
            time = default_timer()-start
            print(total_loss, '\t', time)

            # check_point = {
            #     'epoch': epoch,
            #     'state_dict': 
            # }



if __name__=="__main__":
    train()