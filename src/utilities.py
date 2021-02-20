# torch modules
import torch
from torch import nn
from torchvision import transforms, datasets, utils

# other modules
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import math
import os

# own modules
from constants import *

def load_generator(g_path):
    """Function for loading the generator."""

    # check if exists
    if (os.path.exists(g_path)):
        G = torch.load(g_path, map_location='cpu')
        G.eval()

        return G
    else:
        raise Exception('Invalid path to generator.')



def load_GAN(d_path, g_path):
    """Function for loading GAN."""

    # check if exists
    if (os.path.exists(d_path) and os.path.exists(g_path)):
        D = torch.load(d_path, map_location='cpu')
        D.eval()

        G = load_generator(g_path)

        return D, G

    else:
        raise Exception('Invalid path to generator or discriminator.')



def load_data(path):
    """Function for loading the training and inpainting dataset.

    The function loads the data from the relevant directories and applies
    transformations. The image is turned into a tensor and normalised.

    Input in nested subfolder as this is requirement of DataLoader.
    """

    # apply transformations to images
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # set image folders
    training_data = datasets.ImageFolder(path, transform=transform)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # return dataloaders
    return train_loader




def initialise_weights(m):
    """Weight initalisation of layer to normal distribution.

    Mean = 0 and standard deviation = 0.02. Parameters learned from DCGAN
    paper.

    Learned from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    mean = 0.0
    std = 0.02

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean, std)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0)



def denormalise(image):
    """Method for turning into image with 0-255 colour scale, as tensors to_range
    from -1 to 1."""

    current_range = np.max(image) - np.min(image)
    denormalised = np.array((image - np.min(image)) / float(current_range), dtype=float)
    denormalised = (denormalised * 255)

    # cast to correct data type
    return denormalised.astype('uint8')



def save_img(path, filename, image, range):
    """Function for saving an image."""

    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, filename)
    utils.save_image(image, path, normalize=True, range=range)



def configure_axes():
    """Configuration of Matplotlib plt axes."""

    # configure axes
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')



def save_graph(path, filename):
    """Save graph."""

    # save graph
    print("Saving graph...")
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    plt.close('all')



def plot_and_save_inpainting(c_vector, p_vector, l_vector, epoch_vector, path, epoch):
    """Plot and save inpainting loss graph"""

    print("Plotting graph...")

    # configure graph construct
    plt.figure(figsize=(16,8))
    plt.title("Inpainting Loss over the Epochs", fontsize=18)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)

    # configure max and min x and y
    max_x = len(epoch_vector)
    min_y = math.floor(min(c_vector + p_vector))
    max_y = math.ceil(max(c_vector + p_vector))
    plt.axis([0, max_x, min_y, max_y])
    plt.grid(True)

    # configure axes
    configure_axes()

    # plot with legend
    plt.plot(epoch_vector, c_vector, 'g-', label="Context Loss")
    plt.plot(epoch_vector, p_vector, 'r-', label="Prior Loss")
    plt.plot(epoch_vector, l_vector, 'b-', label="Total Loss")
    plt.legend(prop={'size': 14})

    save_graph(path, "inpaintingGraph_afterEpoch{}.png".format(epoch))


def plot_and_save(epoch_vector, dy_vector, gy_vector, epoch, path, cy_vector=None):
    """Plot and save loss graph.

    Function for plotting and saving a graph visualising the discriminator
    and generator losses across the epochs.
    """

    print("Plotting graph...")

    # configure graph construct
    plt.figure(figsize=(16,8))
    plt.title("GAN Loss Function over the Epochs", fontsize=20)
    plt.ylabel('Loss ', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)

    # configure max and min x and y
    max_x = epoch_vector[-1]

    # check how many networks to plot
    if cy_vector is not None:
        min_y = math.floor(min(dy_vector + cy_vector))
        max_y = math.ceil(max(dy_vector + cy_vector))
    else:
        min_y = math.floor(min(dy_vector + gy_vector))
        max_y = math.ceil(max(dy_vector + gy_vector))

    plt.axis([0, max_x, min_y, max_y])
    plt.grid(True)

    configure_axes()

    # plot
    plt.plot(epoch_vector, gy_vector, 'r-', label="Generator")
    plt.plot(epoch_vector, dy_vector, 'g-', label="Discriminator")

    if cy_vector is not None:
        plt.plot(epoch_vector, cy_vector, 'b-', label="Critic")

    plt.legend(prop={'size': 16})

    # save graph
    save_graph(path, "epochGraph_afterEpoch{}.png".format(epoch))



def visualise_over_time(epochs, fake_imgs, path):
    """Function for visualising progression of 1 image over epochs."""

    # calc dimensions
    num_imgs = len(fake_imgs)
    cols = 4
    rows = math.ceil(num_imgs / cols)

    # make figure
    plt.figure(figsize=(cols * 6, rows * 6 + 5))

    # configure title
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.suptitle("Progression of an generated Image", fontsize=20)

    axes = np.array(axes)

    idx = 0
    # iterate over images
    for r in range(rows):
        for c in range(cols):
            ax = plt.subplot2grid((rows, cols), (r, c))

            if idx < num_imgs:
                ax.title.set_text("Epoch {}".format(epochs[idx]))
                ax.axis("off")

                # reverse normalisation
                img = fake_imgs[idx] * torch.tensor(0.5).view(1)
                img = img + torch.tensor(0.5).view(1)
                transform_img = transforms.ToPILImage(mode='RGB')
                ax.imshow(transform_img(img.squeeze()).resize((256, 256)))

                idx += 1

            else:
                ax.set_visible(False)

    # save image
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = os.path.join(path, 'outputOverTime.png')
    plt.savefig(full_path)
    plt.close('all')



def visualise(epoch, path, fake_imgs, train_dl, device):
    """Function for visualising real and fake data after certain number of epochs.

    Learned visualisation from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    print("Visualising...")

    # subplot data
    subplot_rows = 2
    subplot_cols = 1
    axes = (subplot_rows, subplot_cols, 0)

    # arrange real images as grid
    real_imgs = next(iter(train_dl))
    real_imgs = utils.make_grid(real_imgs[0].to(device)[:BATCH_SIZE], padding=2, normalize=True).cpu()
    real_imgs = np.transpose(real_imgs, axes)

    fake_imgs = utils.make_grid(fake_imgs, padding=2, normalize=True).cpu()
    fake_imgs = np.transpose(fake_imgs, axes)

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 20), nrows=subplot_rows, ncols=subplot_cols)
    axes = np.array(axes)

    # visualise real images
    ax1.set_title("Real Images", fontsize=20)
    ax1.axis("off")
    ax1.imshow(real_imgs)


    # visualise fake images
    ax2.set_title("Fake Images", fontsize=20)
    ax2.axis("off")
    ax2.imshow(fake_imgs)

    # save image
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = os.path.join(path, 'output_after{}Epoch.png'.format(epoch))
    plt.savefig(full_path)
    plt.close('all')



def save_model(D, G, path, max_epoch, Critic=None):
    """Function for saving generator and discriminator"""

    print("Saving discriminator and generator...")

    # switch to evaluation mode
    D.eval()
    G.eval()

    # construct paths and save
    Path(path).mkdir(parents=True, exist_ok=True)
    d_path = os.path.join(path, "discriminator_epoch{}.pt".format(max_epoch))
    g_path = os.path.join(path, "generator_epoch{}.pt".format(max_epoch))
    torch.save(D, d_path)
    torch.save(G, g_path)

    # save critic if given
    if Critic is not None:
        Critic.eval()
        critic_path = os.path.join(path, "critic_epoch{}.pt".format(max_epoch))
        torch.save(Critic, critic_path)
