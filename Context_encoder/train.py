from model import Context_Encoder, Adversarial_Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import utils
import numpy as np
import pandas as pd



def train(model, load, loss, trainset_path, testset_path, dataset_path, 
          model_path, result_path, pretrained_model_path, save_path, 
          n_iter=10000, lr=0.0001, batch_size=500, lambda_recon=0.999, 
          overlap_size=7, hiding_size=64, device=3, check_path=None):

    if checkPath is not None:
        state = torch.load(checkPath)
        model.load_state_dict(state['state_dict'])

    if not os.path.exists(model_path):
        os.makedirs( model_path )

    if not os.path.exists(result_path):
        os.makedirs( result_path )

    if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):
        imagenet_images = []
        for dir, _, _, in os.walk(dataset_path):
            imagenet_images.extend( glob( os.path.join(dir, '*.jpg')))

        imagenet_images = np.hstack(imagenet_images)

        trainset = pd.DataFrame({'image_path':imagenet_images[:int(len(imagenet_images)*0.9)]})
        testset = pd.DataFrame({'image_path':imagenet_images[int(len(imagenet_images)*0.9):]})

        trainset.to_pickle( trainset_path )
        testset.to_pickle( testset_path )
    else:
        trainset = pd.read_pickle( trainset_path )
        testset = pd.read_pickle( testset_path )

    testset.index = range(len(testset))
    testset = testset.ix[np.random.permutation(len(testset))]


