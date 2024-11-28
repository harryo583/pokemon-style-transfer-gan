import torch
import itertools
import torch
import torch.nn as nn
from networks import ResnetGenerator
from networks import NLayerDiscriminator


class CycleGAN:
    def __init__(self, input_nc, output_nc, device):
        self.device = device
        self.netG_A2B = ResnetGenerator(input_nc, output_nc).to(device)
        self.netG_B2A = ResnetGenerator(output_nc, input_nc).to(device)
        self.netD_A = NLayerDiscriminator(input_nc).to(device)
        self.netD_B = NLayerDiscriminator(output_nc).to(device)

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
