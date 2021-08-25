import torch
import config
from data_preparation import CycleDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from discriminator import Discriminator
from generator import Generator
from utils import save_network, load_network, save_outputs
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('ggplot')


def train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse, epoch):
    samples_root = Path('./Saved_Samples')
    
    loop = tqdm(enumerate(loader), leave=True)
    
    running_disc_loss = 0.0
    running_gen_loss = 0.0

    batch_disc_loss = []
    batch_gen_loss = []

    # X is `Sketch` 
    # Y is `Digital`
    # Generator-X takes in a Digital-Image(y) and cvts it to a Sketch-Image
    # Generator-Y takes in a Sketch-Image(y) and cvts it to a Digital-Image
    for idx, (x, y) in loop:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Training the Discriminators (Adversarial training so we use MSE Loss)

        # Discriminator_X (works on the X-Samples)
        fake_x = gen_X(y)  # Fake sample X generated from the Y->X Generator
        disc_X_real = disc_X(x)  # Discriminator predictions on real X samples
        disc_X_fake = disc_X(fake_x.detach())  # Discriminator predictions on the fake X samples

        disc_X_real_loss = mse(disc_X_real, torch.ones_like(disc_X_real))  # The loss on the real samples by the disc
        disc_X_fake_loss = mse(disc_X_fake, torch.zeros_like(disc_X_fake))  # The loss on the fake samples by the disc
        disc_X_loss = disc_X_real_loss + disc_X_fake_loss

        # Discriminator_Y (works on the Y-Samples)
        fake_y = gen_Y(x)  # Fake sample Y generated from the X->Y Generator
        disc_Y_real = disc_Y(y)  # Discriminator predictions on the real Y samples
        disc_Y_fake = disc_Y(fake_y.detach())  # Discriminator predictions on the fake Y samples

        disc_Y_real_loss = mse(disc_Y_real, torch.ones_like(disc_Y_real))  # The loss on the real samples by the disc
        disc_Y_fake_loss = mse(disc_Y_fake, torch.zeros_like(disc_Y_fake))  # The loss on the fake samples by the disc
        disc_Y_loss = disc_Y_real_loss + disc_Y_fake_loss

        # Putting it together
        disc_loss = (disc_X_loss + disc_Y_loss) / 2

        # Updating the parameters of the discriminators
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Training the Generators (Adversarial training uses MSE and identity-mapping and cycle loss uses L1)
        disc_X_fake = disc_X(fake_x)  # Predictions by the discriminator on the fake X (Computation included)
        disc_Y_fake = disc_Y(fake_y)  # Predictions by the discriminator on the fake Y (Computation included)

        # Adversarial loss for the generators
        gen_X_loss = mse(disc_X_fake, torch.ones_like(disc_X_fake))
        gen_Y_loss = mse(disc_Y_fake, torch.ones_like(disc_Y_fake))

        # Cycle consistent loss
        cycle_X = gen_X(fake_y)
        cycle_Y = gen_Y(fake_x)
        cycle_x_loss = l1(x, cycle_X)
        cycle_y_loss = l1(y, cycle_Y)

        # Identity loss
        identity_x = gen_X(x)
        identity_y = gen_Y(y)
        identity_x_loss = l1(x, identity_x)
        identity_y_loss = l1(y, identity_y)


        # Putting it together
        gen_loss = (
            # Complete Adversarial loss
                gen_X_loss
                + gen_Y_loss
                # Complete Cycle loss
                + cycle_x_loss * config.LAMBDA_CYCLE
                + cycle_y_loss * config.LAMBDA_CYCLE
                # Complete Identity loss
                + identity_x_loss * config.LAMBDA_IDENTITY
                + identity_y_loss * config.LAMBDA_IDENTITY
                # Complete Paired loss
        )

        # Updating the parameters of the generators
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
        samples_path = samples_root / f'Epoch-{epoch}'
        if not samples_path.exists():
            samples_path.mkdir()
            
        # The fake Digital Generated for a real Sketch
        save_outputs(x, fake_y, samples_path / f'RealSketch-FakeDig-{idx}.png')  
        # The fake Sketch Generated for a real Digital
        save_outputs(y, fake_x, samples_path / f'RealDig-FakeSketch-{idx}.png')

        running_disc_loss += disc_loss.item()
        running_gen_loss += gen_loss.item()

        batch_disc_loss.append(disc_loss.item())
        batch_gen_loss.append(gen_loss.item())

        loop.set_description(f'Step [{idx+1}/{len(loader)}]')
        loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

    plt.plot(batch_disc_loss)
    plt.plot(batch_gen_loss)
    plt.title('Batch Wise Loss Plot')
    plt.legend(['Discriminator Loss', 'Generator Loss'])
    plt.tight_layout()
    
    plt.savefig(f'Plots/BatchwiseLoss-{epoch}.png', dpi=600)
    plt.clf()

    return running_disc_loss / len(loader), running_gen_loss / len(loader)


def main():
    print(f'Running on device: {config.DEVICE}')

    disc_losses = []
    gen_losses = []
    
    disc_x = Discriminator(in_channels=3).to(config.DEVICE)
    disc_y = Discriminator(in_channels=3).to(config.DEVICE)
    gen_y = Generator(img_channels=3, n_residuals=9).to(config.DEVICE)
    gen_x = Generator(img_channels=3, n_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_x.parameters()) + list(disc_y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_y.parameters()) + list(gen_x.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = CycleDataset(
        # X-Images are sketches and Y-Images are Digitals
        root_x=config.TRAIN_DIR + "/Sketches", root_y=config.TRAIN_DIR + "/Digitals", transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    load_network('./DiscriminatorX-55.pth.tar', disc_x, opt_disc, lr=config.LEARNING_RATE)
    load_network('./GeneratorX-55.pth.tar', gen_x, opt_gen, lr=config.LEARNING_RATE)
    load_network('./DiscriminatorY-55.pth.tar', disc_y, opt_disc, lr=config.LEARNING_RATE)
    load_network('./GeneratorY-55.pth.tar', gen_y, opt_gen, lr=config.LEARNING_RATE)

    for epoch in range(50):
        disc_loss, gen_loss = train_fn(disc_x, disc_y, gen_y, gen_x, loader, opt_disc, opt_gen, L1, mse, epoch)
        
        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)

        if config.SAVE_MODEL:
            save_network(f'Models/GeneratorX-{epoch}.pth.tar', gen_x, opt_gen)
            save_network(f'Models/GeneratorY-{epoch}.pth.tar', gen_y, opt_gen)
            save_network(f'Models/DiscriminatorX-{epoch}.pth.tar', disc_x, opt_disc)
            save_network(f'Models/DiscriminatorY-{epoch}.pth.tar', disc_y, opt_disc)
    
    plt.plot(disc_losses)
    plt.plot(gen_losses)
    plt.title('Epoch Loss Plot')
    plt.legend(['Discriminator Loss', 'Generator Loss'])
    plt.tight_layout()
    plt.savefig(f'Plots/EpochLoss.png', dpi=600)


if __name__ == "__main__":
    main()
