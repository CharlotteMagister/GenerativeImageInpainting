#
# Implementation of the paper 'Semantic Image Inpainting with Deep Generative Models'
# by Yeh et al.
#

import torch
from torchvision import transforms, utils
from torch import nn, optim
from PIL import Image

from pathlib import Path
import numpy as np
import random
import math
import sys
import os
import cv2

from WGAN import Discriminator, Generator
from constants import *
from utilities import *

def get_mask():
    """Function for constructing a random mask."""

    mask = np.ones((IMG_SIZE, IMG_SIZE))

    # limit randomised mask corners
    small_idx = IMG_SIZE // 4
    large_idx = small_idx * 3

    x = np.random.randint(small_idx, large_idx)
    y = np.random.randint(small_idx, large_idx)
    w = np.random.randint(small_idx, large_idx) // 2
    h = np.random.randint(small_idx, large_idx) // 2

    # ensure witin image bounds
    h_low = max(0, x-h)
    h_high = min(IMG_SIZE, x+h)
    w_low =  max(0, y-w)
    w_high = min(IMG_SIZE, y+w)

    # cut out section of mask
    mask[h_low:h_high, w_low:w_high] = 0

    return mask, ((1,) + mask.shape), (h_low, h_high, w_low, w_high)



def get_imgs(path):
    """Function for getting original and corrupted image."""

    # image shape
    shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

    # load image and perform preprocessing
    org_img = Image.open(path)
    transform = transforms.Compose([
                        transforms.CenterCrop(256),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    org_img = transform(org_img)

    # make mask
    mask, mask_shape, mask_coords = get_mask()

    # cut hole into image to get image to be inpainted
    inpaint_img = org_img.clone().detach().numpy()
    inpaint_img[0][1 - mask > 0.5] = np.max(inpaint_img)
    inpaint_img = torch.FloatTensor(inpaint_img)

    # cast to tensor
    mask = torch.FloatTensor(mask.reshape(mask_shape))

    return org_img, inpaint_img, mask, mask_coords



def calc_loss(inpaint_img, gen_img, mask, pred):
    """Calculate prior and context loss and then derive total loss."""

    scalar = 0.01
    # based on equation in paper
    prior_loss = scalar * torch.sum(-torch.log(pred))
    context_loss = torch.sum(((inpaint_img - gen_img) ** 2) * mask)
    loss = context_loss + prior_loss

    return prior_loss.item(), context_loss.item(), loss


def save_output(org_img, inpaint_img, gen_img, mask_img, path):
    """Method for saving the output as images."""

    # find image range to normalise
    img_range = torch.min(org_img), torch.max(org_img)

    save_img(path, 'original.png', org_img, img_range)
    save_img(path, 'original_hole.png', mask_img, img_range)
    save_img(path, 'generated_img.png', gen_img, img_range)
    save_img(path, 'original_inpainted.png', inpaint_img, img_range)



def train(org_img, inpaint_img, mask, path):
    """Function for training and finding image to inpaint.

    Takes orginal image, image with hole and mask. Then performs gradient descent
    to find best latent vector space z yielding best image to inpaint from GAN.
    """

    D, G = load_GAN(D_PATH, G_PATH)

    # inpainting
    optimal_z = nn.Parameter(torch.randn(1, Z, 1, 1))
    adam_optimiser = optim.Adam([optimal_z], lr=0.1)

    print("Perform gradient descent...")

    # iterations
    max_epoch = 10000
    current_epoch = 0

    # graph output
    prior_losses = []
    context_losses = []
    losses = []

    # book keeping for stopping condition
    prev_loss = sys.maxsize
    loss_diff = prev_loss

    while (current_epoch <= max_epoch) or (loss_diff <= 0.001):

        # set gradients to zero
        adam_optimiser.zero_grad()

        # get generator output and predict
        gen_img = G(optimal_z)
        pred = D(gen_img)

        # calculate loss
        p, c, loss = calc_loss(inpaint_img, gen_img, mask, pred)
        prior_losses.append(p)
        context_losses.append(c)
        losses.append(loss.item())

        # back propagate
        loss.backward()

        # print statistics
        if (current_epoch % 500 == 0):
            print("Epoch {}\tLoss: {:.3f}".format(current_epoch, round(loss.item(), 3)))

        # optimise weights
        adam_optimiser.step()

        # clip values
        for weight in optimal_z:
            weight.data.clamp_(-1, 1)

        # book keeping for stopping condition
        loss_diff = abs(prev_loss - loss)
        prev_loss = loss

        current_epoch += 1

    # visualise
    epoch_vector = list(range(0, current_epoch))
    plot_and_save_inpainting(context_losses, prior_losses, losses, epoch_vector, path, current_epoch)

    return gen_img

def perform_poisson_blending(path, mask_coords):
    """Method for performing possion blending."""

    # load generated image and image inpainted roughly
    src = cv2.imread(path + '/generated_img.png')
    dst = cv2.imread(path + '/original_inpainted.png')

    # enlarge mask to get area to blend
    lh = max([0, mask_coords[0] - 10])
    hh = min([IMG_SIZE, mask_coords[1] + 10])
    lw = max([0, mask_coords[2] - 10])
    hw = min([IMG_SIZE, mask_coords[3] + 10])
    src = src[lh:hh, lw:hw]

    # find center to place src image
    center_h = lh + ((hh - lh) // 2)
    center_w = lw + ((hw - lw) // 2)
    center = (center_w, center_h)

    # make mask in this shape, somehow mask of 0s doesnt work
    mask = 255 * np.ones(src.shape, src.dtype)

    # blend
    blended_img = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)

    # save poisson blending image
    Path(path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.path.join(path, "poisson_blended_image.png"), blended_img)

def perform_blending(path, mask_coords):
    """Method for blending by averaging pixels."""

    # load generated and inpainted image
    src = Image.open(path + '/generated_img.png')
    dst = Image.open(path + '/original_inpainted.png')

    # cast to alpha
    org = Image.new('RGBA', size=(IMG_SIZE, IMG_SIZE), color=(0, 0, 0, 0))
    org.paste(dst, (0, 0))

    # cast to alpha
    overlay = Image.new('RGBA', size=(64, 64), color=(0, 0, 0, 0))
    overlay.paste(src, (0,0))

    # save image
    result = Image.blend(org, overlay, alpha=0.3)
    result.save(path + "/blended.png")



def inpaint(org_img, gen_img, mask, path, mask_coords):
    """Function for performing inpainting and saving images."""

    print("Performing inpainting...")

    # get image with hole
    hole_img = mask * org_img
    # get inpainted image
    basic_inpaint_img = mask * org_img + (1 - mask) * gen_img

    # save output
    save_output(org_img, basic_inpaint_img, gen_img, hole_img, path)

    # perform blending
    perform_poisson_blending(path, mask_coords)
    perform_blending(path, mask_coords)



def main():
    """Semantic image inpainting using generator of previously trained WGAN."""

    global device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NUM_GPUS > 0) else "cpu")

    # load image for inpainting
    inpaint_img_path = sys.argv[1]

    # save output
    output_path = "../output/inpainting/newResults"

    # get original, inpainting and mask img
    org_img, inpaint_img, mask, mask_coords = get_imgs(inpaint_img_path)

    # find best image to inpaint
    gen_img = train(org_img, inpaint_img, mask, output_path)

    # inpaint
    inpaint(org_img, gen_img, mask, output_path, mask_coords)



if __name__ == "__main__":
    main()
