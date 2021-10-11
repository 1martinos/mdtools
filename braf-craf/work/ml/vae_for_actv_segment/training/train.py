import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py as h
from datetime import datetime as dt
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import embed as e
import torch.cuda.amp as amp
torch.set_num_threads(4)
from cvae import cVAE as CVAE
from torch.cuda.amp import autocast 
"""
First we are trying with unaltered distances and a default
CVAE as defined by Shozeb's team!
"""

# COMMAND LINE ARGS ########################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("epochs",
                    help="Number of Epochs to train for.",
                    type=int)
parser.add_argument("latent",
                    help="Dimension of the AE latent space.",
                    type=int)
args = parser.parse_args()
###########################################################


def vae_loss(recon_x, x, mu, logvar, reduction='mean'):
        BCE = F.binary_cross_entropy_with_logits(recon_x,x,reduction="mean")
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD.sum(1).mean(0, True)
        return BCE, KLD

def get_cur_date():
    cur = str(dt.now())
    cur = cur.replace(" ","_")[2:].rsplit(":",1)[0]
    cur = cur.replace(":","")
    return cur

if __name__ == '__main__':
    cur_date = get_cur_date()

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Hyper-parameters
    EPOCHS = args.epochs
    l_r = 0.00001
    d_l = args.latent # Latent dimension
    data_path = "../data/training_data.h5"
    data = h.File(data_path,"r")
    train_data = data["train/data"]
    test_data = data["test/data"]
    eval_data = data["validation/data"]
    num_samples, *dsize = train_data.shape
    print(f"Train Data: {num_samples}")
    print(f"Test Data: {len(test_data)}")
    print(f"Val Data: {len(eval_data)}")
    print(f"Data Shape: {dsize}")
    print(f"Training for {EPOCHS}")
    print(f"Latent Space dimension: {d_l}")

    batch_size = int(len(train_data)/200)
    cvae = CVAE(dsize,d_l)
    gscaler = amp.GradScaler(enabled=True)
    if device.type == "cuda":
        pin = True 
        cvae.to(device)
        print("Running on GPU")
        print(torch.cuda.get_device_name(0))
    else: 
        pin = False

    optim = torch.optim.RMSprop(cvae.parameters(), 
                                lr=l_r,
                                alpha=0.9,eps=1e-08)
    dataLoader = DataLoader(train_data, batch_size=batch_size,
                            pin_memory=pin,shuffle=True)
    dataLoader_eval = DataLoader(eval_data, batch_size=batch_size,
                            pin_memory=pin,shuffle=True)
    file_dir = f"../models/{EPOCHS}epochs-{d_l}latent"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
        os.makedirs(file_dir+"/progress")
    print("Starting Training...")
    loss_list = []
    for epoch in tqdm(range(EPOCHS),desc="Epoch",leave=False):
        t1 = time.time()
        loss_per_Epoch = 0
        BCE_per_Epoch = 0
        KLD_per_Epoch = 0
        for i, batch in enumerate(tqdm(dataLoader,leave=False,desc="Batch")):
            optim.zero_grad()
            if device.type == "cuda":
                batch = batch.to(device)
            batch = batch.float().view(-1,1,*dsize)
            with autocast():
                reconstruct_x, mean, log_var, z = cvae(batch)
                BCE,KLD = vae_loss(reconstruct_x,
                                 batch.view(-1,*dsize),
                                 mean,
                                 log_var)
                loss = BCE + KLD
            loss_list.append(
                f"EPOCH: {epoch} BATCH: {i} BCE: {BCE.item()} " 
                f"KLD: {KLD.item()} EFF_LOSS: {loss.item()}"
                )

            gscaler.scale(loss).backward()
            gscaler.step(optim)
            gscaler.update()

            loss_per_Epoch += loss.item()
            BCE_per_Epoch += BCE.item()
            KLD_per_Epoch += KLD.item()
            del batch
        t2 = time.time()
        loss_eval = 0
        with torch.no_grad():        # Test for each Epoch
            with open(f"./{file_dir}/progress/{epoch}.p","wb") as f:
                for batch_eval in dataLoader_eval:
                    if device.type == "cuda":
                        batch_eval = batch_eval.to(device)
                    batch_eval = batch_eval.float().view(-1,1,*dsize)
                    reconstruct_x, mean, log_var, z = cvae(batch_eval)
                    BCE,KLD = vae_loss(reconstruct_x,
                                 batch_eval.view(-1,*dsize),
                                 mean,
                                 log_var)
                    loss_eval += BCE + KLD
                    pickle.dump((reconstruct_x.cpu(),
                                 batch_eval.cpu()),
                                 f)
                with open(f"./{file_dir}/test_loss.txt","a") as f:
                    f.write(f"{epoch}    {loss_eval.item()}")
                    f.write("\n")
        del batch_eval, loss_eval
        #print('GPU Memory Usage:',end="\r")
        #print('Allocated:',round(torch.cuda.memory_allocated(0)/1024**3,1),'GB',end="\r")
        #print('Cached:   ',round(torch.cuda.memory_reserved(0)/1024**3,1),'GB', end="\r")
        #print(f"Loss for epoch {epoch}: {round(loss_per_Epoch, 3)}",end="\r")
        #print(f"BCE: {round(BCE_per_Epoch, 3)}  KLD: {round(KLD_per_Epoch,3)}",end="\r")
    pickle.dump(cvae,open(f"{file_dir}/{EPOCHS}epochs-{d_l}dim.pickle", "wb"))
    with open(f"./{file_dir}/loss_stats.txt","w") as f:
        for x in loss_list:
            f.write(x + "\n")
