import sys
import torch.nn as nn
import argparse
import surfa as sf
import scipy.ndimage
import torch
from models.model import UNet
from PIL import Image
import numpy as np
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.image import math_img, load_img
import nilearn as nl
from nilearn import plotting as nplot
import os
import shutil
from glob import glob
import cv2
import nibabel as nib
from tqdm.notebook import tqdm
import torch

def main(args) :

    print(f'step 1. get model')
    model = UNet()

    print(f' (1.2) loading pretrained model')
    version = '1'
    modelfile = '/home/dreamyou070/pretrained_model/'
    checkpoint = torch.load(modelfile, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()


    print(f'step 2. inference with model')

    file = 'NFBS/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz'
    image = sf.load_volume(file)

    """
    # predict the surface distance transform
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_np).to(args.device)
        output = model(input_tensor, 1)
        mni_norm = output[0].cpu().numpy().squeeze().astype(np.int16)
        norm = output[1].cpu().numpy().squeeze().astype(np.int16)
        scalar_field = output[2].cpu().numpy().squeeze().astype(np.int16)

    # unconform the sdt and extract mask
    mni_norm = conformed.new(mni_norm).resample_like(image, method='nearest', fill=0)
    norm = conformed.new(norm).resample_like(image, method='nearest', fill=0)
    scalar_field = conformed.new(scalar_field).resample_like(image, method='nearest', fill=0)
    """

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda', )
    args = parser.parse_args()
    main(args)