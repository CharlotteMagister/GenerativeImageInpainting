#
# WGAN based on original paper: "Wasserstein GAN" by Martin Arjovsky et al.
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

    Simple copy of DCGAN generator with added layer.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Generator, self).__init__()
        n_in = Z
        n_out = IMG_CHANNELS

        feature_map = IMG_SIZE_LARGE
        kernel_size = 4
        stride = 2
        padding = 1
        bias = False

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # input is latent variable space Z
            nn.ConvTranspose2d(n_in, feature_map * 16, kernel_size, 1, 0, bias=bias),
            nn.BatchNorm2d(feature_map * 16),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 16
            nn.ConvTranspose2d(feature_map * 16, feature_map * 8, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 8
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 4
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(inplace=True),

            # nodes = feature_map * 2
            nn.ConvTranspose2d(feature_map * 2, feature_map, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(inplace=True),

            # nodes = 1
            nn.ConvTranspose2d(feature_map, n_out, kernel_size, stride, padding, bias=bias),
            nn.Tanh()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)



class Discriminator(nn.Module):
    """Class definining the Critic.

    WGAN architecture. The class contains the definition of the network architecture.
    WGAN takes an image and outputs a probability of the image being true.

    Simple copy of DCGAN discriminator with added layer and Sigmoid function removed.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Discriminator, self).__init__()
        n_in = IMG_CHANNELS
        n_out = 1

        feature_map = IMG_SIZE_LARGE
        kernel_size = 4
        stride = 2
        padding = 1
        bias = False

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # nodes = 128 x 128 x 3
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
            nn.Conv2d(feature_map * 8, feature_map * 16, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = 1
            nn.Conv2d(feature_map * 16, n_out, kernel_size, 1, 0, bias=bias),
            # scratched sigmoid activation function
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)



class Discriminator2(nn.Module):
    """Class definining the Discriminator.

    DCGAN architecture. The class contains the definition of the network architecture.
    DCGAN takes an image and outputs a probability of the image being true.

    Simple copy of DCGAN discriminator with added layer.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Discriminator2, self).__init__()
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

            # nodes =  feature_map * 4
            nn.Conv2d(feature_map * 2, feature_map * 4, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 8
            nn.Conv2d(feature_map * 4, feature_map * 8, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = feature_map * 16
            nn.Conv2d(feature_map * 8, feature_map * 16, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(feature_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # nodes = 1
            nn.Conv2d(feature_map * 16, n_out, kernel_size, 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        return self.network(input)



def train_discriminator(real_data, batch_size):
    """Logic for training the discriminator."""

    # set network gradients to 0
    d_solver.zero_grad()

    # train on real data
    prediction_r = D(real_data).view(-1)

    # train on fake data
    noise = torch.randn(batch_size, Z, 1, 1, device=device)
    fake_data = G(noise)
    prediction_f = D(fake_data.detach()).view(-1)

    # perform back propagation
    loss = -torch.mean(prediction_r) + torch.mean(prediction_f)
    loss.backward()

    # adjust weights
    d_solver.step()

    # weight clipping to stay in function def
    c = 0.01
    for w in D.parameters():
        w.data.clamp_(-c, c)

    return loss, prediction_r.mean().item(), prediction_f.mean().item()


def train_generator(batch_size):
    """Logic for training the generator"""

    # reset gradients
    g_solver.zero_grad()

    # predict on fake data
    noise = torch.randn(batch_size, Z, 1, 1, device=device)
    prediction = D(G(noise)).view(-1)

    # perform back propagation
    loss = -torch.mean(prediction)
    loss.backward()

    # adjust weights
    g_solver.step()

    return loss, prediction.mean().item()



def train_discriminator2(batch_data, batch_size):
    """Logic for training the discriminator."""

    target_r = torch.full((batch_size,), SMOOTH_REAL_LABEL, device=device)
    target_f = torch.full((batch_size,), FAKE_LABEL, device=device)
    noise = torch.randn(batch_size, Z, 1, 1, device=device)

    # set network gradients to 0
    D2.zero_grad()

    # train on real data
    prediction_r = D2(batch_data).view(-1)
    error_r = BCELossFunc(prediction_r, target_r)

    # train on fake data
    fake_data = G(noise)
    prediction_f = D2(fake_data.detach()).view(-1)
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
    train_dl = load_data(PATH_TO_LARGE_TRAINING_DATA)

    # epochs
    max_epoch = 200000
    current_epoch = 0

    # output for plotting
    dy_vector = []
    gy_vector = []
    d2y_vector = []
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
            de_vector = []
            for _ in range(1):
                d_error, dr_prediction, df_prediction = train_discriminator(batch_data, batch_size)
                de_vector.append(d_error)

            d_error = sum(de_vector) / len(de_vector)

            # train generator
            ge_vector = []
            for _ in range(2):
                g_error, g_prediction = train_generator(batch_size)
                ge_vector.append(g_error)

            g_error = sum(ge_vector) / len(ge_vector)

            # normal discriminator
            d2_error = train_discriminator2(batch_data, batch_size)

            # save data
            if current_epoch % 20 == 0:
                dy_vector.append(d_error)
                gy_vector.append(g_error)
                d2y_vector.append(d2_error)
                epoch_vector.append(current_epoch)

            # record progress
            if (current_epoch != 0) and (current_epoch % 10000 == 0):
                # print progress
                print("Epoche {} - D Loss: {} \t G Loss: {}".format(current_epoch, round(d_error.item(), 3), round(g_error.item(), 3)))

                # plot progress
                plot_and_save(epoch_vector, d2y_vector, gy_vector, current_epoch, path, dy_vector)

                # visualise progress
                with torch.no_grad():
                    img = G(test_noise).detach().cpu()
                    img2 = G(vis_noise).detach().cpu()

                fake_img_over_time.append(img2)
                fake_img_epoch.append(current_epoch)
                visualise(current_epoch, path, img, train_dl, device)

            current_epoch += 1

            if current_epoch > max_epoch:
                # ensure leave batch iter loop
                break

    visualise_over_time(fake_img_epoch, fake_img_over_time, path)

    return max_epoch


def main():
    """Main function driving training of WGAN."""

    global G
    global D
    global D2

    global device

    global d_solver
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
    D = Discriminator(NUM_GPUS).to(device)
    D.apply(initialise_weights)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        D = nn.DataParallel(D, list(range(NUM_GPUS)))

    # Discriminator
    D2 = Discriminator2(NUM_GPUS).to(device)
    D2.apply(initialise_weights)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        D2 = nn.DataParallel(D2, list(range(NUM_GPUS)))

    # loss function and optimisers
    d_solver = optim.RMSprop(D.parameters(), lr=1e-4)
    g_solver = optim.RMSprop(G.parameters(), lr=1e-4)

    # loss function and optimisers
    BCELossFunc = nn.BCELoss()
    d_optimiser = optim.Adam(D2.parameters(), lr=0.0002, betas=(0.5, 0.999))

    path = "../output/WGAN_128px/newResults"

    epochs = train(path)

    save_model(D2, G, path, epochs, D)


if __name__ == "__main__":
    main()
