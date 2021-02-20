#
# Unrolled GAN modelled on paper "Unrolled Generative Adversarial Networks" by Luke Metz et al.
#

# pytorch modules required
import torch
from torch import nn, optim
from torchvision import transforms, datasets, utils

# other modules
import numpy as np
import random
import math
import copy
import os

# own modules
from utilities import *
from constants import *

# using Discrimiantor and Generator form DCGAN
from DCGAN import Generator, Discriminator

def train_discriminator(batch_data, batch_size):
    """Logic for training the discriminator."""

    # desired values
    target_r = torch.full((batch_size, 1, 1, 1), SMOOTH_REAL_LABEL, device=device)
    target_f = torch.full((batch_size, 1, 1, 1), FAKE_LABEL, device=device)
    noise = torch.randn(batch_size, Z, 1, 1, device=device)

    # set network gradients to 0
    d_optimiser.zero_grad()

    # train on real data
    prediction_r = D(batch_data)
    error_r = loss(prediction_r, target_r)

    # train on fake data
    fake_data = G(noise)
    prediction_f = D(fake_data.detach())
    error_f = loss(prediction_f, target_f)

    # perform back propagations
    error = error_r + error_f
    error.backward()

    # adjust weights
    d_optimiser.step()

    return error, prediction_r.mean().item(), prediction_f.mean().item()



def train_generator(train_dl, batch_size):
    """Logic for training the generator."""

    # number of times unroll performed
    num_unrolls = 5

    # target
    target_r = torch.full((batch_size, 1, 1, 1), SMOOTH_REAL_LABEL, device=device)
    target_f = torch.full((batch_size, 1, 1, 1), FAKE_LABEL, device=device)

    # reset gradients
    G.zero_grad()

    # do not want to use real copy
    D_copy = copy.deepcopy(D)
    copy_optimiser = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # real data iterator
    batch_data_iter = iter(train_dl)

    # perfrom unrolling step
    for _ in range(num_unrolls):
        # reset gradients
        copy_optimiser.zero_grad()

        # get real data and predict, loop over batch if necessary
        try:
            batch_data = batch_data_iter.next()
        except StopIteration:
            batch_data_iter = iter(train_dl)
            batch_data = batch_data_iter.next()

        batch_data = batch_data[0].to(device)
        batch_size2 = batch_data.size(0)

        # get fake data
        noise = torch.randn(batch_size2, Z, 1, 1, device=device)
        with torch.no_grad():
            fake_imgs = G(noise)

        # predict
        prediction_r = D_copy(batch_data)
        prediction_f = D_copy(fake_imgs)

        # perform back propagation
        error = loss(prediction_r, target_r) + loss(prediction_f, target_f)
        error.backward(create_graph=True)

        # adjust weights
        copy_optimiser.step()

    # generate fake data and predict
    noise = torch.randn(batch_size, Z, 1, 1, device=device)
    prediction = D_copy(G(noise))

    # perform backwards propagation
    error = loss(prediction, target_f)
    error.backward()

    # adjust weights
    g_optimiser.step()

    # clean up
    del D_copy

    return error, prediction.mean().item()



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
                g_error, g_prediction = train_generator(train_dl, batch_size)
                ge_vector.append(g_error)

            g_error = sum(ge_vector) / len(ge_vector)

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

                fake_imgs = fake_imgs.view(fake_imgs.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
                visualise(current_epoch, path, fake_imgs, train_dl, device)

                vis_img = vis_img.view(vis_img.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
                fake_img_over_time.append(vis_img)
                fake_img_epoch.append(current_epoch)

            current_epoch += 1

            if current_epoch > max_epoch:
                # ensure leave batch iter loop
                break

    visualise_over_time(fake_img_epoch, fake_img_over_time, path)

    return max_epoch


def main():
    """Main function driving training of Vanilla GAN."""

    # define random seed to allow reporducibility - random.randint(1, 10000)
    seed = 97
    random.seed(seed)
    torch.manual_seed(seed)

    global G
    global D

    global device

    global loss
    global d_optimiser
    global g_optimiser

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

    # loss function and optimisers
    loss = nn.BCELoss()
    d_optimiser = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    g_optimiser = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

    path = "../output/unrolledGAN/newResults"

    max_epoch = train(path)

    save_model(D, G, path, max_epoch)


if __name__ == "__main__":
    main()
