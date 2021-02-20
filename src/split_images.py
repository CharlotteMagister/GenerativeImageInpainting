from pathlib import Path
import cv2
import os

def tile_images():
    """Method for splitting images to derive processed dataset."""

    # tiles sizes
    big_tile_size = 256
    small_tile_size = 64
    t = 0

    # iterate over images
    # for submission using the evaluation image submitted to keep number of files low
    # actual experiments used separate batch of imgs
    for root, dirs, files in os.walk("../input/evaluationData/."):
        for file in files:
            path = os.path.join(root, file)
            if (path.endswith('.png')):
                print(path)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED);

                # slide tiles over image
                for h in range(0, img.shape[0] - big_tile_size, big_tile_size):
                    for w in range(0, img.shape[1] - big_tile_size, big_tile_size):

                        # get tile and resize to needed size
                        tile = img[h:h+big_tile_size,w:w+big_tile_size]
                        tile = cv2.resize(tile, (small_tile_size, small_tile_size), interpolation = cv2.INTER_AREA)

                        black_pixels = 0

                        # calculate how much useful of images (not black)
                        height, width, depth = tile.shape
                        for i in range(0, height):
                            for j in range(0, width):
                                px_sum = 0
                                for k in range(0, depth):
                                    px_sum += tile[i][j][k]

                                if px_sum < 10:
                                    black_pixels += 1

                        # check ratio and save tile
                        ratio = black_pixels / (small_tile_size * small_tile_size)
                        print("Ratio ", ratio)
                        if ratio < 0.25:
                            path = "../input/newResults/"
                            Path(path).mkdir(parents=True, exist_ok=True)
                            path = os.path.join(path, "tile{}.png".format(t))

                            print("Saving tile...", path)
                            cv2.imwrite(path, tile)
                            t += 1

tile_images()
