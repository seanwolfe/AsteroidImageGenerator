import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, ManualInterval, AsinhStretch, ContrastBiasStretch, HistEqStretch, \
    LogStretch, LinearStretch, PowerStretch, PowerDistStretch, SinhStretch, SqrtStretch, SquaredStretch
from astropy.utils.data import get_pkg_data_filename
import os
import random
import matplotlib.patches as patches
import shutil
from PIL import Image
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
import yaml


#  open stack
def open_stack(stack_folder, stack_file_names):
    """
    Opens the image by first using get_random_file() to obtain a file path. Splits the file into a header portion
    and an image portion. The input dimensions of the image is extracted from the header. Pre processing tasks,
    thresholding and stretching are applied to get post processed image.
    :return:
    """

    stack_files = []

    for idx, stack in enumerate(stack_file_names):
        stack_file = os.path.join(stack_folder, stack)
        if stack_file:
            # open .fits file
            image_file = get_pkg_data_filename(stack_file)
            # print image info
            fits.info(image_file)
            # get data from image (ie pixel values)
            image_data = fits.getdata(image_file, ext=0)
            header = fits.getheader(image_file, ext=0)
            image_dimensions = (header['NAXIS1'], header['NAXIS2'])
            stack_files.append([image_data, image_dimensions])
        else:
            raise ValueError("No files found in the directory.")

    return stack_files


def random_crop(stack_files, targets_data, output_size, padding=10):
    """
    Crop an image according to output dimensions, while ensuring that the output dimensions of the image are
    satisfied. Calculate the region of allowable starting crop pixel locations, and then randomly chose from
    these locations in order to crop an image of specified dimensions. Stor in image_crop property and crop_start
    property for location.
    :return:
    """

    output_length = output_size[0]
    output_height = output_size[1]
    stack_crops = []
    for idx, target in targets_data.iterrows():
        crops = []
        target_id = target['Asteroid ID']
        target_tlc_x = target['LOW X']
        target_tlc_y = target['LOW Y']
        target_length = target['HIGH X'] - target['LOW X']
        target_height = target['HIGH Y'] - target['LOW Y']
        done = 0
        for jdx, stack_file in enumerate(stack_files):

            if done == 0:
                image_data = stack_file[0]
                image_dimensions = stack_file[1]
                length = image_dimensions[0]
                height = image_dimensions[1]

                min_length = target_tlc_x + target_length - output_length
                min_height = target_tlc_y + target_height - output_height
                max_length = target_tlc_x + output_length
                max_height = target_tlc_y + output_height

                length_start = max(0, min_length)
                height_start = max(0, min_height)
                length_end = min(length, max_length)
                height_end = min(height, max_height)

                min_x = length_start + padding
                max_x = length_end - output_length - padding
                min_y = height_start + padding
                max_y = height_end - output_height - padding
                print(target_id)
                print(min_x)
                print(max_x)
                print(min_y)
                print(max_y)
                crop_start_x = np.random.randint(min_x, max_x)
                crop_start_y = np.random.randint(min_y, max_y)

                image_crop = image_data[crop_start_y:crop_start_y + output_height, crop_start_x:crop_start_x + output_length]
                crops.append([(crop_start_x, crop_start_y), image_crop])

                done = 1
            else:
                image_data = stack_file[0]
                image_crop = image_data[crop_start_y:crop_start_y + output_height,
                             crop_start_x:crop_start_x + output_length]
                crops.append(image_crop)
        stack_crops.append([target_id, (crop_start_x, crop_start_y), crops])

    return stack_crops



target_data_1_16 = pd.read_csv('real_tracklet_pixels_1-16.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
target_data_17_32 = pd.read_csv('real_tracklet_pixels_17-32.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
target_data_33_48 = pd.read_csv('real_tracklet_pixels_33-48.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
stack_folder = os.path.join('synthetic_tracklets', 'real_image_stacks', 'ds3 (1)', 'ds3_c_a')
stack_file_names = [f'{i}.fit' for i in range(1, 49)]

stack = open_stack(stack_folder, stack_file_names[:16])
output_size = (224, 224)
stack_crops = random_crop(stack, target_data_1_16, output_size, padding=10)
print(stack_crops)

