from model import Context_Encoder
import torch
import numpy as np


def inference(img, check_point, hole_size=64, device=1):
    img = img.astype(np.float32)
    # img = np.transpose(img, [2, 0, 1]).astype(np.float32)
    # img = np.expand_dims(img, 0)
    # img = np.vstack((img, img)) # set 2 same img to avoid batch_norm-related problems
    img = torch.from_numpy(img)
    N, C, H, W = img.shape

    state = torch.load(check_point)

    ce = Context_Encoder()
    ce.load_state_dict(state['state_dict_c'])

    if device is not None:
        ce = ce.cuda(device)
        img = img.cuda(device)
    
    mask_recon = np.ones(shape=(hole_size, hole_size), dtype=np.float32)
    mask_all = np.pad(mask_recon, (128-hole_size)//2, "constant")
    mask_all = np.expand_dims(mask_all, 0)
    mask_all = np.concatenate([mask_all]*3, 0)
    mask_context = 1-mask_all
    mask_context = np.expand_dims(mask_context, 0)
    mask_context = np.vstack([mask_context]*32)
    mask_context = torch.from_numpy(mask_context)


    if device is not None:
        mask_context = mask_context.cuda(device)

    print(img.type())
    print(mask_context.type())
    img_feed = img*mask_context
    img_hole = ce(img_feed)

    if device is not None:
        img_feed = img_feed.cpu()
        img_hole = img_hole.cpu()


    img_feed = img_feed.numpy()
    img_feed = np.squeeze(img_feed)
    img_feed = np.transpose(img_feed, [0,2,3,1])


    img_hole = img_hole.detach().numpy()
    img_hole = np.squeeze(img_hole)
    img_hole = np.transpose(img_hole, [0,2,3,1])


    img_inpaint = img_feed.copy()
    print(img_inpaint.shape)
    print(img_feed.shape)
    for i in range(3):
        img_inpaint[:,64-hole_size//2:64+hole_size//2, 64-hole_size//2:64+hole_size//2, i] = \
            img_inpaint[:,64-hole_size//2:64+hole_size//2, 64-hole_size//2:64+hole_size//2, i]+img_hole[:, :, :, i]

    img_feed = normalize(img_feed)
    img_inpaint = normalize(img_inpaint)

    return img_feed, img_inpaint


def normalize(img):
    img = img-img.min()
    img = img/(img.max()-img.min())
    img = np.round(img*255)
    img = img.astype(np.uint8)
    return img