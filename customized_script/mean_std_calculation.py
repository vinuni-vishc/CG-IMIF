import numpy as np 
import matplotlib.pyplot as plt
import os
import pdb
import albumentations as A
import cv2
import torch

from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2

class PillData(Dataset):
    
    def __init__(self, 
                 directory, 
                 transform = None):
        self.directory = directory
        self.transform = transform

        self.data = []

        for path_specify in ['train', 'test']:
            path_data = os.path.join(self.directory, path_specify)
            for image_name in os.listdir(path_data):
                path_img = os.path.join(path_data, image_name)
                self.data.append(path_img)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # import
        path  = self.data[idx]
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        return image

if __name__ == "__main__":
    image_size = 256

    augs = A.Compose([A.Resize(height = image_size, 
                            width  = image_size),
                    A.Normalize(mean = (0, 0, 0),
                                std  = (1, 1, 1)),
                    ToTensorV2()])

    # dataset
    image_dataset = PillData(directory = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Large/',
                            transform = augs)

    # data loader
    image_loader = DataLoader(image_dataset, 
                            batch_size  = 8, 
                            shuffle     = False, 
                            num_workers = 2,
                            pin_memory  = True)

    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    total_num_image = 0
    for inputs in tqdm(image_loader):
        total_num_image += inputs.shape[0]
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    ####### FINAL CALCULATIONS
    pdb.set_trace()

    # pixel count
    count = total_num_image * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    # train only
    # results Pill Base
    # mean: tensor([0.4550, 0.5239, 0.5653])
    # std: tensor([0.2460, 0.2446, 0.2252])

    # results Pill Large
    # mean: tensor([0.4807, 0.5434, 0.5801])
    # std:  tensor([0.2443, 0.2398, 0.2222])      


    # train and test
    # results Pill Base
    # mean: tensor([0.5047, 0.5651, 0.5921])
    # std:  tensor([0.2593, 0.2481, 0.2315])

    # results Pill Large
    # mean: tensor([0.4798, 0.5429, 0.5804])
    # std:  tensor([0.2444, 0.2397, 0.2218])  

