import numpy as np
import cv2
import math
import scipy.stats
import sys

# own modules
from utilities import *
from constants import *

from WGAN import Discriminator, Generator

def get_eval_data(G):
    """Load real images and generate images for evaluation"""

    eval_dl = load_data(PATH_TO_TRAINING_DATA)
    batch_iter = iter(eval_dl)

    # real images
    real_batch = next(batch_iter)[0]
    batch_size = real_batch.size(0)

    # fake images
    test_noise = torch.randn(batch_size, Z, 1, 1)
    fake_batch = G(test_noise).detach().cpu()

    return real_batch, fake_batch, batch_size


def calc_PSNR(real_batch, fake_batch, batch_size):
    """Calculate PSNR for batch of real and fake images."""

    PSNR_total = 0

    # iterate over batch
    for i in range(0, batch_size):

        # preprocessing so same colour range
        real_img = denormalise(real_batch[i].detach().numpy())
        fake_img = denormalise(fake_batch[i].detach().numpy())

        # calc difference and find mean squared error
        img_diff = real_img - fake_img
        mean_square_error = np.mean(img_diff ** 2)

        # if no noise then 100
        psnr = 100

        # PSNR calc
        if mean_square_error > 0:
            mean_error = math.sqrt(mean_square_error)
            psnr = 20 * math.log10(255 / mean_error)

        PSNR_total += psnr

    # average
    PSNR = PSNR_total / batch_size

    print("PSNR: ", round(PSNR, 3))



def calc_SNR(real_batch, fake_batch, batch_size):
    """Calculate SNR for batch of real and fake images."""

    SNR_real = 0
    SNR_fake = 0

    # iterate over batch of images
    for i in range(0, batch_size):

        # cast back to 0-255 colour range
        real_img = denormalise(real_batch[i].flatten().detach().numpy())
        fake_img = denormalise(fake_batch[i].flatten().detach().numpy())

        # learned from: https://www.codespeedy.com/calculate-signal-to-noise-ratio-in-python/

        # axis for taking mean since image
        axis = 0

        # calculate mean
        real_mean = np.mean(real_img, axis)
        fake_mean = np.mean(fake_img, axis)

        # calculate standard deviation
        real_std = np.std(real_img, axis)
        fake_std = np.std(fake_img, axis)

        # calculate SNR
        SNR_real += np.where(real_std == 0, 0, real_mean / real_std)
        SNR_fake += np.where(fake_std == 0, 0, fake_mean / fake_std)

        # end learned from

    SNR_real_avg = SNR_real / batch_size
    SNR_fake_avg = SNR_fake / batch_size
    print("Average SNR of real images: ", round(SNR_real_avg, 3))
    print("Average SNR of fake images: ", round(SNR_fake_avg, 3))




def calc_accuracy(D, real_batch, fake_batch, batch_size):
    """Calculate accuracy of discriminator for classifying
    a batch of real and fake images."""

    total = batch_size * 2
    correct_pred = 0

    # get list of predicted values for batch
    pred_r = D(real_batch).view(-1).tolist()
    # check predition, should be 1
    for pred in pred_r:
        if pred > 0.5:
            correct_pred += 1

    # get list of predicted values for batch
    pred_f = D(fake_batch).view(-1).tolist()
    # check prediction, should be 0
    for pred in pred_f:
        if pred <= 0.5:
            correct_pred += 1

    accuracy = correct_pred / total
    print("Accuracy: ", round(accuracy, 3))

def print_usage():
    """Method for printing usage statement.

    Currently not used as no longer used to generate batch of images.
    """

    print("There are two modes. The '-generate' mode generates sample images. The '-evaluate' mode evaluates a model.")
    print("Usage 1: python3 evaluate.py -generate <path_to_generator> <output_path>")
    print("Usage 2: python3 evaluate.py -evaluate <path_to_generator> <path_to_discriminator>")
    print("Note evaluation model is specific to WGAN model.")

def generate_sample_images(g_path, output_path):
    """Method for generating sample images."""

    # load generator and sample training data
    G = load_generator(g_path)
    train_dl = load_data(PATH_TO_TRAINING_DATA)

    # create noise to create fake images
    test_noise = torch.randn(BATCH_SIZE, Z, 1, 1, device='cpu')
    single_img_noise = torch.randn(1, Z, 1, 1, device='cpu')

    # get fake images
    imgs = G(test_noise).detach().cpu()
    single_img = G(single_img_noise).detach().cpu()
    range = torch.min(single_img), torch.max(single_img)

    # visualise and save
    visualise("Last", output_path, imgs, train_dl, 'cpu')
    save_img(output_path, "single_img.png", single_img, range)

def evaluate_model(g_path, d_path):
    """Method for driving evaluation of model."""

    D, G = load_GAN(d_path, g_path)

    real_batch, fake_batch, batch_size = get_eval_data(G)

    calc_PSNR(real_batch, fake_batch, batch_size)
    calc_SNR(real_batch, fake_batch, batch_size)
    calc_accuracy(D, real_batch, fake_batch, batch_size)

def main():
    """Main function driving evaluation of model by calculating PSNR, SNR and
    accuracy for batch of real and fake images.

    Additional functionalities for generating images using model.
    """

    evaluate_model(G_PATH, D_PATH)

    #num_arg = len(sys.argv)
    # # parse arguments
    # if num_arg == 4:
    #     flag = sys.argv[1]
    #
    #     try:
    #         if (flag == '-generate'):
    #             generate_sample_images(sys.argv[2], sys.argv[3])
    #
    #         elif (flag == '-evaluate'):
    #             evaluate_model(sys.argv[2], sys.argv[3])
    #
    #         else:
    #             print_usage()
    #     except Exception as e:
    #         print(repr(e))
    #         print_usage()
    # else:
    #     print_usage()



if __name__ == "__main__":
    main()
