import torch
import torch.nn as nn

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        for _ in range(2):
            model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(128),
                      nn.ReLU(True)]
        for _ in range(n_blocks):
            model += [ResnetBlock(128)]
        for _ in range(2):
            model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(True)]
        model += [nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, True)]
        for n in range(1, n_layers):
            nf_mult_prev = min(2 ** (n - 1), 8)
            nf_mult = min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                 kernel_size=4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(ndf * nf_mult),
                       nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
