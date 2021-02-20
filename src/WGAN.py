#
# WGAN based on original paper: "Wasserstein GAN" by Martin Arjovsky et al.
# Reusing implementation of DCGAN mostly. Tweaks to design learned from paper.
#

# pytorch modules required
import torch
from torch import nn, optim
from torchvision import transforms, datasets, utils

# other modules
import numpy as np
import random
import math
import os

# own modules
from utilities import *
from constants import *

class Generator(nn.Module):
    """ Generator network definition.

    WGAN GAN. Network takes latent variable vector of size 100 as input and outputs an
    image. For this, each layer scales up the number of nodes by a factor of 8.
    The penultimate layer scales above the number of nodes needed and then scales
    back down.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Generator, self).__init__()
        n_in = Z
        n_out = IMG_CHANNELS

        feature_map = IMG_SIZE
        kernel_size = 4
        stride = 2
        padding = 1
        bias = False

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # input is latent variable space Z
            nn.ConvTranspose2d(n_in, feature_map * 8, kernel_size, 1, 0, bias=bias),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 4
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 2
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(inplace=True),

            # nodes = feature_map
            nn.ConvTranspose2d(feature_map * 2, feature_map, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(inplace=True),

            # nodes = output image size
            nn.ConvTranspose2d(feature_map, n_out, kernel_size, stride, padding, bias=bias),
            nn.Tanh()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)



class Critic(nn.Module):
    """Class definining the critic, which is the network playing against the discriminator.

    WGAN architecture. The class contains the definition of the network architecture.
    WGAN takes an image and outputs a probability of the image being true.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Critic, self).__init__()
        n_in = IMG_CHANNELS
        n_out = 1

        feature_map = IMG_SIZE
        kernel_size = 4
        stride = 2
        padding = 1
        bias = False

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # nodes = IMG_CHANNELS * IMG_SIZE * IMG_SIZE
            nn.Conv2d(n_in, feature_map, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 2
            nn.Conv2d(feature_map, feature_map * 2, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 4
            nn.Conv2d(feature_map * 2, feature_map * 4, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 8
            nn.Conv2d(feature_map * 4, feature_map * 8, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 8
            nn.Conv2d(feature_map * 8, n_out, kernel_size, 1, 0, bias=bias),
            # scratched sigmoid activation function
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)


class Discriminator(nn.Module):
    """Class definining the discriminator. This discriminator is not used in the game
    against the generator, but is needed for inpainting.

    DCGAN architecture. The class contains the definition of the network architecture.
    DCGAN takes an image and outputs a probability of the image being true.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Discriminator, self).__init__()
        n_in = IMG_CHANNELS
        n_out = 1

        feature_map = IMG_SIZE
        kernel_size = 4
        stride = 2
        padding = 1
        bias = False

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # input is image
            nn.Conv2d(n_in, feature_map, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 2
            nn.Conv2d(feature_map, feature_map * 2, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 4
            nn.Conv2d(feature_map * 2, feature_map * 4, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 8
            nn.Conv2d(feature_map * 4, feature_map * 8, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = 1
            nn.Conv2d(feature_map * 8, n_out, kernel_size, 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)



def train_critic(real_data, batch_size):
    """Logic for training the critic."""

    # set network gradients to 0
    c_solver.zero_grad()

    # train on real data
    prediction_r = C(real_data).view(-1)

    # train on fake data
    noise = torch.randn(batch_size, Z, 1, 1, device=device)
    fake_data = G(noise)
    prediction_f = C(fake_data.detach()).view(-1)

    # perform back propagation
    # implemenation of loss learned from Pytorch functionality learned from https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
    loss = -torch.mean(prediction_r) + torch.mean(prediction_f)
    loss.backward()

    # adjust weights
    c_solver.step()

    # weight clipping to stay in function def
    # Pytorch functionality learned from https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
    c = 0.01
    for weight in C.parameters():
        weight.data.clamp_(-c, c)

    return loss, prediction_r.mean().item(), prediction_f.mean().item()


def train_generator(batch_size):
    """Logic for training the generator"""

    # reset gradients
    g_solver.zero_grad()

    # predict on fake data
    noise = torch.randn(batch_size, Z, 1, 1, device=device)
    prediction = C(G(noise)).view(-1)

    # perform back propagation
    # implemenation of loss learned from Pytorch functionality learned from https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
    loss = -torch.mean(prediction)
    loss.backward()

    # adjust weights
    g_solver.step()

    return loss, prediction.mean().item()

def train_discriminator(batch_data, batch_size):
    """Logic for training the discriminator."""

    target_r = torch.full((batch_size,), SMOOTH_REAL_LABEL, device=device)
    target_f = torch.full((batch_size,), FAKE_LABEL, device=device)
    noise = torch.randn(batch_size, Z, 1, 1, device=device)

    # set network gradients to 0
    d_optimiser.zero_grad()

    # train on real data
    prediction_r = D(batch_data).view(-1)
    error_r = BCELossFunc(prediction_r, target_r)

    # train on fake data
    fake_data = G(noise)
    prediction_f = D(fake_data.detach()).view(-1)
    error_f = BCELossFunc(prediction_f, target_f)

    # perform back propagations
    error = error_r + error_f
    error.backward()

    # adjust weights
    d_optimiser.step()

    return error

def train(path):
    """ Function undertaking the training of the generator and discriminator.

    Nested for loop for iterating over the epochs and batches respectively and
    training the generator and discriminator.
    """

    print("Starting training...")

    # import training data
    train_dl = load_data(PATH_TO_TRAINING_DATA)

    # epochs
    max_epoch = 80000
    current_epoch = 0

    # output for plotting
    cy_vector = []
    gy_vector = []
    dy_vector = []
    epoch_vector = []

    fake_img_over_time = []
    fake_img_epoch = []

    test_noise = torch.randn(BATCH_SIZE, Z, 1, 1, device=device)
    vis_noise = torch.randn(1, Z, 1, 1, device=device)

    while (current_epoch <= max_epoch):

        # mini-batches best for stochastic gradient descent
        for batch_num, batch_data in enumerate(train_dl):

            # format batch
            batch_data = batch_data[0].to(device)
            batch_size = batch_data.size(0)

            # train discriminator
            ce_vector = []
            for _ in range(1):
                c_error, cr_prediction, cf_prediction = train_critic(batch_data, batch_size)
                ce_vector.append(c_error)

            c_error = sum(ce_vector) / len(ce_vector)

            # normal discriminator
            d_error = train_discriminator(batch_data, batch_size)

            # train generator
            ge_vector = []
            for _ in range(2):
                g_error, g_prediction = train_generator(batch_size)
                ge_vector.append(g_error)

            g_error = sum(ge_vector) / len(ge_vector)

            # save data
            if current_epoch % 20 == 0:
                cy_vector.append(c_error)
                gy_vector.append(g_error)
                dy_vector.append(d_error)
                epoch_vector.append(current_epoch)

            # record progress
            if (current_epoch != 0) and (current_epoch % 10000 == 0):
                # print progress
                print("Epoche {} - C Loss: {} \t G Loss: {}".format(current_epoch, round(c_error.item(), 3), round(g_error.item(), 3)))

                # plot progress
                plot_and_save(epoch_vector, dy_vector, gy_vector, current_epoch, path, cy_vector)

                # visualise progress
                with torch.no_grad():
                    img = G(test_noise).detach().cpu()
                    img2 = G(vis_noise).detach().cpu()

                fake_img_over_time.append(img2)
                fake_img_epoch.append(current_epoch)
                visualise(current_epoch, path, img, train_dl, device)

                save_model(D, G, path, current_epoch, C)

            current_epoch += 1

            if current_epoch > max_epoch:
                # ensure leave batch iter loop
                break

    visualise_over_time(fake_img_epoch, fake_img_over_time, path)

    return max_epoch


def main():
    """Main function driving training of WGAN."""

    global G
    global C
    global D

    global device

    global c_solver
    global g_solver
    global d_optimiser

    global BCELossFunc

    # define random seed to allow reporducibility
    seed = 97
    torch.manual_seed(seed)
    random.seed(seed)

    # optimise for GPU learned from Vanilla GAN tutorial:
    # https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NUM_GPUS > 0) else "cpu")

    # Generator
    G = Generator(NUM_GPUS).to(device)
    G.apply(initialise_weights)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        G = nn.DataParallel(G, list(range(NUM_GPUS)))

    # Discriminator
    C = Critic(NUM_GPUS).to(device)
    C.apply(initialise_weights)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        C = nn.DataParallel(C, list(range(NUM_GPUS)))

    # Discriminator
    D = Discriminator(NUM_GPUS).to(device)
    D.apply(initialise_weights)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        D = nn.DataParallel(D, list(range(NUM_GPUS)))

    # loss function and optimisers as in DCGAN paper
    BCELossFunc = nn.BCELoss()
    d_optimiser = optim.Adam(D.parameters(), lr=1e-4)

    # loss function and optimisers
    c_solver = optim.RMSprop(C.parameters(), lr=1e-4)
    g_solver = optim.RMSprop(G.parameters(), lr=1e-4)

    path = "../output/WGAN/newResults"

    epochs = train(path)

    # last parameter is optional for saving critic
    save_model(D, G, path, epochs, C)


if __name__ == "__main__":
    main()
