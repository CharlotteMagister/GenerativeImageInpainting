import numpy as np
import imutils
import cv2
import sys
import csv


def stitch():
    """Function driving execution."""
    if (len(sys.argv) != 5):
        print("Usage statement: python stitching.py img1 img2 pointFile1 pointFile2")
        return

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_UNCHANGED)
    keyPts1 = parse_key_points(sys.argv[3])
    keyPts2 = parse_key_points(sys.argv[4])

    #adjust_brightness(img1, img2)
    homographyMatrix, masked = align_images(keyPts1, keyPts2)

    perform_stitching(img1, img2, keyPts1, keyPts2, homographyMatrix)



def adjust_brightness(img1, img2):
    """Attempt at smoothing colours."""
    hsvImg = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsvImg)
    s = s * 1.3
    s = np.clip(s,0,255)
    hsvImg = cv2.merge([h,s,v])

    out = cv2.cvtColor(hsvImg.astype("uint8"), cv2.COLOR_HSV2BGR)

    cv2.imwrite("adjusted2.png", out)



def get_mask(img):
    """Make mask of image."""
    height, width = img[0], img[1]
    mask = numpy.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            if (img[3] != 0):
                mask[y, x] = 1
    return mask



def convert_to_grey_scale(image):
    """Convert image to greyscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def parse_key_points(fileName):
    """Parses key points from file to list."""
    listOfPoint2f = []

    with open(fileName, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for line in reader:
            listOfPoint2f.append(cv2.KeyPoint(np.float32(line[0]), np.float32(line[1]), 1))

    return np.array(listOfPoint2f)[:5]



def align_images(keyPts1, keyPts2):
    """Function for aligning the images."""
    keyPts1 = np.float32([p.pt for p in keyPts1])
    keyPts2 = np.float32([p.pt for p in keyPts2])

    if (len(keyPts1) >= 4) & (len(keyPts2) >= 4):
        homographyMatrix, masked = cv2.findHomography(keyPts1, keyPts2, 0)
        return (homographyMatrix, masked)
    else:
        raise AssertionError("Not enough keypoints")



def perform_stitching(imageA, imageB, keyPts1, keyPts2, H):
    """Function performing the transformation and actual stitching.

    Learned from: https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
    """

    height, width = (imageA.shape[0] + imageB.shape[0]), (imageA.shape[1] + imageB.shape[1])
    result = cv2.warpPerspective(imageB, H, (width, height))
    #cv2.imwrite("resultIntermediate1.png", result)

    channels = 4
    imageInter = np.zeros((height, width, channels), dtype=np.uint8)
    imageInter[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    #cv2.imwrite("resultIntermediate2.png", imageInter)

    for y in range(0, height):
        for x in range(0, width):
            pixelA = result[y,x]
            pixelB = imageInter[y,x]
            first option
            if (pixelA[3] == 0) & (pixelB[3] != 0):
                print("case 1")
                result[y,x][0] = pixelB[0]
                result[y,x][1] = pixelB[1]
                result[y,x][2] = pixelB[2]
                result[y,x][3] = pixelB[3]
            elif (pixelA[3] != 0) & (pixelB[3] != 0):
                intermediate = np.array([pixelA, pixelB])
                result[y,x] = np.copy(np.median(intermediate, axis=0))

                print("Height " + str(height) + " Width " + str(width) + " Current " + str(y) + " " + str(x))
                result[y,x][0] = (pixelA[0] + pixelB[0]) / 2
                result[y,x][1] = (pixelA[1] + pixelB[1]) / 2
                result[y,x][2] = (pixelA[2] + pixelB[2]) / 2
                result[y,x][3] = (pixelA[3] + pixelB[3]) / 2

            second options
            if (pixelA[3] == 0) & (pixelB[3] != 0):
                print("case 1")
                result[y,x][0] = pixelB[0]
                result[y,x][1] = pixelB[1]
                result[y,x][2] = pixelB[2]
                result[y,x][3] = pixelB[3]
            elif (pixelA[3] != 0) & (pixelB[3] != 0):
                if (((y > 0) & (x > 0)) & ((y < (height-2)) & (x < (width-2)))):
                    print("Height " + str(height) + " Width " + str(width) + " Current " + str(y) + " " + str(x))
                    intermediate = np.array([pixelA, pixelB, result[y-1,x], result[y+1,x], result[y,x-1], result[y,x+1], result[y-1,x-1], result[y+1,x+1], result[y-1,x+1], result[y+1,x-1], imageInter[y-1,x], imageInter[y+1,x], imageInter[y,x-1], imageInter[y,x+1], imageInter[y-1,x-1], imageInter[y+1,x+1], imageInter[y-1,x+1], imageInter[y+1,x-1]])
                    result[y,x] = np.copy(np.median(intermediate, axis=0))

    cv2.imwrite("recent.png", result)


    cv2.imwrite("resultIntermediate.png", result)
    img_width, img_height = (imageA.shape[0] + imageB.shape[0]), (imageA.shape[1] + imageB.shape[1])
    n_channels = 4
    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
    transparent_img[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    result = cv2.addWeighted(result,1,transparent_img,1,0)

    cv2.imwrite("result.png", result)

# execution
stitch()
