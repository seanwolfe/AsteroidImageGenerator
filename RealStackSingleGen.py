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
def open_region(stack_folder, stack_file_names, regions, output_size):
    """
    Opens the image by first using get_random_file() to obtain a file path. Splits the file into a header portion
    and an image portion. The input dimensions of the image is extracted from the header. Pre processing tasks,
    thresholding and stretching are applied to get post processed image.
    :return:
    """
    output_length = output_size[0]
    output_height = output_size[1]
    target_stack_files = []
    for jdx, row in regions.iterrows():
        stack_files = []
        done = 0
        for idx, stack in enumerate(stack_file_names):
            stack_file = os.path.join(stack_folder, stack)
            if stack_file:
                # open .fits file
                print(stack_file)
                image_file = get_pkg_data_filename(stack_file)
                # print image info
                fits.info(image_file)
                # get data from image (ie pixel values)
                image_data = fits.getdata(image_file, ext=0)
                if done == 0:
                    min_x = row['LOW X']
                    max_x = row['HIGH X']
                    min_y = row['LOW Y']
                    max_y = row['HIGH Y']
                    crop_start_x = np.random.randint(min_x, max_x)
                    crop_start_y = np.random.randint(min_y, max_y)
                    done = 1

                image_crop = image_data[crop_start_y:crop_start_y + output_height,
                             crop_start_x:crop_start_x + output_length]
                if row['Target'] == '2018 RB':
                    image_crop[:60, :60] = image_crop[60:120, :60].copy()
                stack_files.append(image_crop)
            else:
                raise ValueError("No files found in the directory.")
        target_stack_files.append([row['Target'], stack_files])

    return target_stack_files


def random_crop(stack_files, output_size):
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
    for idx, stack in enumerate(stack_files):
        crops = []
        done = 0
        for jdx, stack_file in enumerate(stack[1]):
            if done == 0:
                image_data = stack_file
                length = len(image_data[0])
                height = len(image_data)

                min_x = 0
                max_x = length - output_length
                min_y = 0
                max_y = height - output_height
                crop_start_x = np.random.randint(min_x, max_x)
                crop_start_y = np.random.randint(min_y, max_y)

                image_crop = image_data[crop_start_y:crop_start_y + output_height, crop_start_x:crop_start_x + output_length]
                crops.append(image_crop)

                done = 1
            else:
                image_data = stack_file
                image_crop = image_data[crop_start_y:crop_start_y + output_height,
                             crop_start_x:crop_start_x + output_length]
                crops.append(image_crop)
        stack_crops.append([stack[0], crops])

    return stack_crops


def process(image_data, row, configuration):

    # preprocessing - interval and stretching, normalize [0,1]
    mean = row[0]
    standard_deviation = row[2]
    stretch = AsinhStretch(a=0.9999)
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
        processed_stack = []
        for jdx, image_crop in enumerate(image_crops):
            sigma_clip = SigmaClip(sigma=configuration['sigmaclip_sigma'],
                                   maxiters=configuration['sigmaclip_maxiters'])
            threshold = detect_threshold(image_crop, nsigma=configuration['detect_threshold_nsigma'],
                                         sigma_clip=sigma_clip)
            segment_img = detect_sources(image_crop, threshold, npixels=configuration['detect_sources_npixels'])
            if segment_img is not None:
                footprint = circular_footprint(radius=configuration['circular_footprint_radius'])
                mask = segment_img.make_source_mask(footprint=footprint)
                row = sigma_clipped_stats(image_crop, sigma=configuration['sigma_clipped_stats_sigma'], mask=mask)
                processed_image_crop = process(image_crop, row, configuration)
                processed_stack.append(processed_image_crop)
                good = True
            else:
                good = False
                break
        if good == True:
            processed_stacks.append([crop[0], processed_stack])
    return processed_stacks


def video_file(final_stacks, low, high):

    final_final_stacks = []
    for idx, final_stack in enumerate(final_stacks):
        target_id = final_stack[0]
        print(target_id)
        final_images = final_stack[1]
        scaled_image_array = (np.array(final_images) * 255).astype(np.uint8)
        final_final_stacks.append(scaled_image_array)
        final_image_array = np.repeat(scaled_image_array[:, :, :, np.newaxis], 3, axis=3)

        # save numpy array as [16, 3, 224, 224]
        final_image_array = np.transpose(final_image_array, (0, 3, 1, 2))

        video_array = final_image_array.copy()
        name = target_id + '_singlerealstack_' + str(low) + str(high)
        np.save(name + '.npy', final_image_array)

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

with open('config_par.yaml', 'r') as f:
    config = yaml.safe_load(f)

stack_folder = os.path.join('synthetic_tracklets', 'real_image_stacks', 'ds3 (1)', 'ds3_c_a')
stack_file_names = [f'{i}.fit' for i in range(0, 61)]
output_size = (224, 224)
regions = pd.read_csv('Databases/regions_1-60_final.csv', sep=',', header=0, names=['Target','LOW X','LOW Y','HIGH X','HIGH Y'])
# 1-17, 17-33, 33-49, 45-61
# redo 33-49
low = 45
high = 61
stack = open_region(stack_folder, stack_file_names[low:high], regions, output_size)

# stack_crops = random_crop(stack, output_size)
processed_stacks = process_stacks(stack, config)
final_stacks = video_file(processed_stacks, low, high)
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