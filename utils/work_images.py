import cv2
from keras.preprocessing import image
import numpy as np
import os
from zipfile import ZipFile


rotated_dict = {
    'rotated_left': 90,
    'rotated_right': 270,
    'upside_down': 180,
    'upright': 0,
}


def read_path(folder_path, size):
    images = []
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)

    return images


def zip_path(path, files):
    with ZipFile(path, 'w') as zip:
        for file in files:

            zip.write(file)


def rotate(image, orientation, save_path):
    global rotated_dict

    img = cv2.imread(image)

    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0

    M = cv2.getRotationMatrix2D(center, rotated_dict['orientation'], scale)
    img = cv2.warpAffine(img, M, (h, w))

    cv2.imwrite(save_path + '/' + image, img)
