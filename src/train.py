import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from cyclegan import CycleGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cyclegan = CycleGAN(3, 3, device)

dataset_A = ImageDataset("/path/to/generic_dataset")
dataset_B = ImageDataset("/path/to/fire_dataset")
dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

epochs = 200
for epoch in range(epochs):
    for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        # Train CycleGAN here
