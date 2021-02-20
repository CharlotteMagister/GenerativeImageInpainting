#
# Vanilla GAN based on the original paper: Generative Adversarial Networks by Ian Goodfellow et al.
# Python functionality learned from: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
#

# pytorch modules required
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import datasets

# other modules
import numpy as np
import random
import math
import os

# own modules
from utilities import *
from constants import *

# define generator
class Generator(torch.nn.Module):
    """ Generator network definition.

    Vanilla GAN. Network takes latent variable vector of size 100 as input and outputs an
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
        n_out = SIZE

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # input layer
            nn.Linear(n_in, IMG_SIZE * 8),
            nn.ReLU(inplace=True),

            # increase number of nodes to 64 * 16
            nn.Linear(IMG_SIZE * 8, IMG_SIZE * 16),
            nn.ReLU(inplace=True),

            # increase number of nodes to 64 * 32
            nn.Linear(IMG_SIZE * 16, IMG_SIZE * 32),
            nn.ReLU(inplace=True),

            # increase number of nodes to 64 * 64
            nn.Linear(IMG_SIZE * 32, IMG_SIZE * 64),
            nn.ReLU(inplace=True),

            # increase number of nodes to 64 * 128
            nn.Linear(IMG_SIZE * 64, IMG_SIZE * 128),
            nn.ReLU(inplace=True),

            # increase number of nodes to 64 * 256
            nn.Linear(IMG_SIZE * 128, IMG_SIZE * 256),
            nn.ReLU(inplace=True),

            # reduce to desired image size
            nn.Linear(IMG_SIZE * 256, n_out),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        # pass through network
        output = self.network(input)
        # format as image
        output = output.view(output.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

        return output


class Discriminator(torch.nn.Module):
    """Class definining the discriminator.

    The class contains the definition of the network architecture.
    """

    def __init__(self, num_gpus):
        """Init function defining constants and network architecture.

        Function definition includes network variable which defines the network
        sequentially.
        """

        super(Discriminator, self).__init__()
        n_in = SIZE
        n_out = 1

        self.num_gpus = num_gpus

        self.network = nn.Sequential(
            # input layer takes images
            nn.Linear(n_in, int(n_in / 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 2), int(n_in / 4)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 4), int(n_in / 8)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 8), int(n_in / 16)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 16), int(n_in / 32)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 32), int(n_in / 64)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # reduce by factor of 2 to previous layer
            nn.Linear(int(n_in / 64), int(n_in / 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # hidden layer dropping to 1 node
            nn.Linear(int(n_in / 128), n_out),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Method for performing forward pass in network."""

        # reshape input
        input = Variable(input.view(input.size(0), SIZE))
        return self.network(input)


def train_discriminator(real_data, batch_size):
    """Logic for training the discriminator."""

    # target
    target_r = torch.full((batch_size,), SMOOTH_REAL_LABEL, device=device)
    target_f = torch.full((batch_size,), FAKE_LABEL, device=device)

    # reset gradients
    d_optimiser.zero_grad()

    # train on real data
    prediction_r = D(real_data).view(-1)
    error_r = loss(prediction_r, target_r)

    # train on fake data
    noise = torch.randn(batch_size, Z, device=device)
    fake_data = G(noise)
    prediction_f = D(fake_data.detach()).view(-1)
    error_f = loss(prediction_f, target_f)

    # perform backpropagation
    error = error_r + error_f
    error.backward()

    # adjust weights
    d_optimiser.step()

    # Return error and predictions for real and fake inputs
    return error, prediction_r.mean().item(), prediction_f.mean().item()



# training the generator
def train_generator(batch_size):
    """Logic for training the generator"""

    # reset gradients
    g_optimiser.zero_grad()

    # target and fake data
    target = torch.full((batch_size,), SMOOTH_REAL_LABEL, device=device)
    noise = Variable(torch.randn(batch_size, Z, device=device))
    fake_data = G(noise)

    # make prediction
    prediction = D(fake_data).view(-1)

    # calculate error
    error = loss(prediction, target)
    error.backward()

    # adjust weights
    g_optimiser.step()

    return error, prediction



# actual training
def train(path):
    """ Function undertaking the training of the generator and discriminator.

    Nested for loop for iterating over the epochs and batches respectively and
    training the generator and discriminator.
    """

    print("Starting training...")

    # import training data
    train_dl = load_data(PATH_TO_TRAINING_DATA)

    # epochs
    max_epoch = 20000
    current_epoch = 0

    # output for plotting
    dy_vector = []
    gy_vector = []
    epoch_vector = []

    fake_img_over_time = []
    fake_img_epoch = []

    test_noise = torch.randn(BATCH_SIZE, Z, device=device)
    vis_noise = torch.randn(1, Z, device=device)

    # outter epoch iterator
    while (current_epoch <= max_epoch):

        # mini-batches best for stochastic gradient descent
        for batch_num, batch_data in enumerate(train_dl):

            # extract images
            batch_data = batch_data[0].to(device)
            batch_size = batch_data.size(0)

            # train the discriminator
            d_error, dr_prediction, df_prediction = train_discriminator(batch_data, batch_size)

            g_error, gr_prediction = train_generator(batch_size)

            # save data
            if current_epoch % 20 == 0:
                dy_vector.append(d_error)
                gy_vector.append(g_error)
                epoch_vector.append(current_epoch)

            # record progress
            if (current_epoch != 0) and (current_epoch % 1000 == 0):
                # print progress
                print("Epoche {} - D Loss: {} \t G Loss: {}".format(current_epoch, round(d_error.item(), 3), round(g_error.item(), 3)))

                # plot progress
                plot_and_save(epoch_vector, dy_vector, gy_vector, current_epoch, path)

                # visualise progress
                with torch.no_grad():
                    fake_imgs = G(test_noise).detach().cpu()
                    vis_img = G(vis_noise).detach().cpu()

                fake_img_over_time.append(vis_img)
                fake_img_epoch.append(current_epoch)
                visualise(current_epoch, path, fake_imgs, train_dl, device)

            current_epoch += 1

            if current_epoch > max_epoch:
                # ensure leave batch iter loop
                break

    visualise_over_time(fake_img_epoch, fake_img_over_time, path)

    return max_epoch


def main():
    """Main function driving training of Vanilla GAN."""

    global G
    global D

    global device

    global loss
    global d_optimiser
    global g_optimiser

    # define random seed to allow reporducibility
    seed = 97
    random.seed(seed)
    torch.manual_seed(seed)

    # optimise for GPU learned from Vanilla GAN tutorial:
    # https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NUM_GPUS > 0) else "cpu")

    # Generator
    G = Generator(NUM_GPUS).to(device)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        G = nn.DataParallel(G, list(range(NUM_GPUS)))

    # Discriminator
    D = Discriminator(NUM_GPUS).to(device)
    if (device.type == 'cuda') and (NUM_GPUS > 1):
        D = nn.DataParallel(D, list(range(NUM_GPUS)))

    # loss function and optimisers
    loss = nn.BCELoss()

    #d_optimiser = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimiser = optim.SGD(D.parameters(), lr=0.01, momentum=0.3)
    g_optimiser = optim.SGD(G.parameters(), lr=0.01, momentum=0.3)

    path = "../output/VanillaGan/newResults"

    max_epoch = train(path)

    save_model(D, G, path, max_epoch)


if __name__ == "__main__":
    main()
