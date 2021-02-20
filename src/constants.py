#
# File defines constants which are the same between the different GANs.
# These constants define aspects of the dataset, such as the image size
# and batch size.
#

IMG_SIZE = 64
IMG_SIZE_LARGE = 128
IMG_CHANNELS = 3
BATCH_SIZE = 64

SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS
SIZE_LARGE = IMG_SIZE_LARGE * IMG_SIZE_LARGE * IMG_CHANNELS

Z = 100

# define paths to images
PATH_TO_TRAINING_DATA = "../input/processed_input_64px"
PATH_TO_LARGE_TRAINING_DATA = "../input/processed_input_128px"
PATH_TO_EVAL_DATA = "../input/evaluationData"

REAL_LABEL = 1
SMOOTH_REAL_LABEL = 0.9
FAKE_LABEL = 0

NUM_GPUS = 1

D_PATH = "../output/WGAN/discriminator_epoch80000.pt"
G_PATH = "../output/WGAN/generator_epoch80000.pt"
