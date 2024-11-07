import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, ManualInterval, AsinhStretch, MinMaxInterval
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
import cv2
import scipy



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


def random_crop(stack_files, targets_data, output_size, padding=0):
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
                crop_start_x = np.random.randint(min_x, max_x)
                crop_start_y = np.random.randint(min_y, max_y)

                image_crop = image_data[crop_start_y:crop_start_y + output_height, crop_start_x:crop_start_x + output_length]
                crops.append(image_crop)


                # cropstart_j = (crop_start_x, crop_start_y)
                #
                # print(target_id)
                # fig = plt.figure()
                # ax = fig.add_subplot(1, 1, 1)

                # Create a rectangle patch
                # rect = patches.Rectangle(xy=cropstart_j, width=224,
                #                          height=224, linewidth=2, edgecolor='r', facecolor='none')
                # Add the rectangle patch to the axis
                # ax.add_patch(rect)

                # im = ax.imshow(image_data, cmap='gray')
                # fig.colorbar(im)
                # plt.show()

                done = 1
            else:
                image_data = stack_file[0]
                image_crop = image_data[crop_start_y:crop_start_y + output_height,
                             crop_start_x:crop_start_x + output_length]
                crops.append(image_crop)
        stack_crops.append([target_id, crops])

    return stack_crops


def process(image_data, row, configuration):

    # preprocessing - interval and stretching, normalize [0,1]
    mean = row[0]
    standard_deviation = row[2]
    stretch = AsinhStretch(a=0.5)
    # clip bright pixels according to percentage of max pixel value
    cut = ManualInterval(mean - configuration['sigma_cutoff_bottom'] * standard_deviation, mean + configuration['sigma_cutoff_top'] * standard_deviation)
    clipped_data = cut(image_data)
    # cut = MinMaxInterval()
    # clipped_data = cut(image_data)
    # stretch data
    # return clipped_data
    return stretch(clipped_data)


def process_stacks(crops, configuration):
    """
    Estimate the background statistics of the image using photoutils package. Clip the image to dampen bright
    sources. Then detect all sources above a certain limit and mask them. Then estimate the mean, median and
    standard deviation of image with masked sources.
    :return:
    """

    processed_stacks = []
    for idx, crop in enumerate(crops):
        image_crops = crop[1]
        target_id = crop[0]
        processed_stack = []
        for jdx, image_crop in enumerate(image_crops):
            sigma_clip = SigmaClip(sigma=configuration['sigmaclip_sigma'],
                                   maxiters=configuration['sigmaclip_maxiters'])
            threshold = detect_threshold(image_crop, nsigma=configuration['detect_threshold_nsigma'],
                                         sigma_clip=sigma_clip)
            segment_img = detect_sources(image_crop, threshold, npixels=configuration['detect_sources_npixels'])
            footprint = circular_footprint(radius=configuration['circular_footprint_radius'])
            mask = segment_img.make_source_mask(footprint=footprint)
            row = sigma_clipped_stats(image_crop, sigma=configuration['sigma_clipped_stats_sigma'], mask=mask)
            processed_image_crop = process(image_crop, row, configuration)
            processed_stack.append(processed_image_crop)
            # print(np.max(image_crop))
            # print(np.min(image_crop))
            # print(np.max(processed_image_crop))
            # print(np.min(processed_image_crop))
        processed_stacks.append([target_id, processed_stack])
    return processed_stacks


def video_file(final_stacks):

    final_final_stacks = []
    for idx, final_stack in enumerate(final_stacks):
        target_id = final_stack[0]
        print(final_stack[0])
        final_images = final_stack[1]
        scaled_image_array = (np.array(final_images) * 255).astype(np.uint8)
        final_final_stacks.append(scaled_image_array)
        final_image_array = np.repeat(scaled_image_array[:, :, :, np.newaxis], 3, axis=3)

        np.save(target_id + '_realstack.npy', final_image_array)
        video_array = final_image_array.copy()
        name = target_id + '_realstack'

        # Define the codec and create VideoWriter object
        num_frames, height, width, channels = video_array.shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec used to compress the frames
        video_filename = name + '.avi'
        video_out = cv2.VideoWriter(video_filename, fourcc, 2.0, (width, height))

        # Write each frame to the video
        for i in range(num_frames):
            frame = video_array[i]
            video_out.write(frame)  # Write the frame

        # Release everything when job is finished
        video_out.release()

        return final_final_stacks

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

target_data_1_16 = pd.read_csv('real_tracklet_pixels_1-16.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
target_data_17_32 = pd.read_csv('real_tracklet_pixels_17-32.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
target_data_33_48 = pd.read_csv('real_tracklet_pixels_33-48.csv', sep=',', header=0, names=['Asteroid ID', 'LOW X', 'MID X', 'HIGH X', 'LOW Y', 'MID Y', 'HIGH Y'])
stack_folder = os.path.join('synthetic_tracklets', 'real_image_stacks', 'ds3 (1)', 'ds3_c')
stack_file_names = [f'{i}.fit' for i in range(1, 49)]

stack = open_stack(stack_folder, stack_file_names[33:49])
output_size = (224, 224)
stack_crops = random_crop(stack, target_data_33_48, output_size, padding=0)
processed_stacks = process_stacks(stack_crops, config)
final_stacks = video_file(processed_stacks)
# final_stacks = video_file(stack_crops)
"""
for jdx, stack_file_j in enumerate(stack):
    image_data_j = stack_file_j[0]
    image_dimensions_j = stack_file_j[1]
    stack_crop_j = stack_crops[jdx]
    target_id_j = stack_crop_j[0]
    # cropstart_j = stack_crop_j[1]
    crops_j = stack_crop_j[1]
    crop_i = crops_j[jdx]

    print(target_id_j)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Create a rectangle patch
    # rect = patches.Rectangle(xy=cropstart_j, width=224,
    #                          height=224, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle patch to the axis
    # ax.add_patch(rect)

    im = ax.imshow(image_data_j, cmap='gray')
    fig.colorbar(im)
    plt.show()

    # circle3 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
    #                          linewidth=1)
    #
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(1, 1, 1)
    # im3 = ax3.imshow(frame1, cmap='gray')
    # ax3.set_title('Summed Signal Image 0 to 1')
    # ax3.add_patch(circle3)
    # Remove both major and minor tick labels
    # ax3.tick_params(axis='both', which='both', bottom=False, top=False,
    #                 left=False, right=False, labelbottom=False, labelleft=False)
"""