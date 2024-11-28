import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from cyclegan import CycleGAN

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CycleGAN
cyclegan = CycleGAN(3, 3, device)

# Datasets
print("Step 1: loading datasets...")
dataset_A = ImageDataset("data/images")
dataset_B = ImageDataset("data/images_by_type/fire")
dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)
print("Step 1 done!")

# Training hyperparameters
epochs = 200
lambda_cycle = 10.0  # weight for cycle consistency loss
lambda_identity = 0.5  # weight for identity loss
batch_limit = 20  # set to limit batches or None to disable

# Training loop
print("Step 2: training loop starting...")
for epoch in range(epochs):
    for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
        if batch_limit and i >= batch_limit:
            break  # stop training after max_batches_per_epoch batches
        
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        ###################
        # Train Generators
        ###################
        cyclegan.set_requires_grad([cyclegan.netD_A, cyclegan.netD_B], False)
        cyclegan.optimizer_G.zero_grad()

        # Identity loss
        loss_idt_A = cyclegan.criterion_identity(cyclegan.netG_B2A(real_B), real_B) * lambda_cycle * lambda_identity
        loss_idt_B = cyclegan.criterion_identity(cyclegan.netG_A2B(real_A), real_A) * lambda_cycle * lambda_identity

        # GAN loss
        fake_B = cyclegan.netG_A2B(real_A)
        loss_GAN_A2B = cyclegan.criterion_GAN(cyclegan.netD_B(fake_B), torch.ones_like(cyclegan.netD_B(fake_B)))

        fake_A = cyclegan.netG_B2A(real_B)
        loss_GAN_B2A = cyclegan.criterion_GAN(cyclegan.netD_A(fake_A), torch.ones_like(cyclegan.netD_A(fake_A)))

        # Cycle consistency loss
        rec_A = cyclegan.netG_B2A(fake_B)
        loss_cycle_A = cyclegan.criterion_cycle(rec_A, real_A) * lambda_cycle

        rec_B = cyclegan.netG_A2B(fake_A)
        loss_cycle_B = cyclegan.criterion_cycle(rec_B, real_B) * lambda_cycle

        # Total generator loss
        loss_G = loss_idt_A + loss_idt_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        cyclegan.optimizer_G.step()

        #######################
        # Train Discriminators
        #######################
        cyclegan.set_requires_grad([cyclegan.netD_A, cyclegan.netD_B], True)
        cyclegan.optimizer_D.zero_grad()

        # Discriminator A
        loss_D_real_A = cyclegan.criterion_GAN(cyclegan.netD_A(real_A), torch.ones_like(cyclegan.netD_A(real_A)))
        loss_D_fake_A = cyclegan.criterion_GAN(cyclegan.netD_A(fake_A.detach()), torch.zeros_like(cyclegan.netD_A(fake_A)))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        # Discriminator B
        loss_D_real_B = cyclegan.criterion_GAN(cyclegan.netD_B(real_B), torch.ones_like(cyclegan.netD_B(real_B)))
        loss_D_fake_B = cyclegan.criterion_GAN(cyclegan.netD_B(fake_B.detach()), torch.zeros_like(cyclegan.netD_B(fake_B)))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        # Total discriminator loss
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        cyclegan.optimizer_D.step()

    # Print epoch status
    print(f"[Epoch {epoch+1}/{epochs}] Loss_G: {loss_G.item()} Loss_D: {loss_D.item()}")

    # Save model checkpoints
    if (epoch + 1) % 50 == 0:
        torch.save(cyclegan.netG_A2B.state_dict(), f"results/checkpoints/netG_A2B_epoch_{epoch+1}.pth")
        torch.save(cyclegan.netD_B.state_dict(), f"results/checkpoints/netD_B_epoch_{epoch+1}.pth")

print("Step 2 training complete!")
