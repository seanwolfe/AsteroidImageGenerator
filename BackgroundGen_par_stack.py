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
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

plt.style.use(astropy_mpl_style)


class BackgroundGen:
    """
    Class for generating stellar backgrounds given images for eventual use in training AI. With a folder of images
    taken by telescopes of the sky, the class will randomly pick an image from the folder, apply some minor
    preprocessing (thresholding and stretching), randomly crop out a sub-image according to specified output
    dimensions, estimate the background statistics of the image, save it in a folder. The class also allows for
    viewing of original and cropped out images, as well as saves a dataframe containing the original file chosen,
    folder in which crop is saved, the background statistics of crop and the pixel location in the original image
    where the crop began. Images are expected to be .FITS files. The cropped image is saved as 0.jpg and 0.npy. The
    image is for viewing whilst the array is for use.
    """

    def __init__(self, config):
        """

        :param real_image_dir: the head level directory where the real images to be used as crops are located
        :param synthetic_image_dir: the head level directory where the cropped out backgrounds should be stored
        :param interval_threshold: the threshold used in preprocessing
        :param stretch: the stretch factor used in a Asinh stretch function
        :param output_image_width: output image width in pixels
        :param ouput_image_height: input image width in pixels
        """
        self.directory = config['real_image_directory']
        self.image_file_path = None  # the file path of the random real image chosen
        self.interval_thresh = config['interval_threshold']
        self.stretch_factor = config['stretch']
        self.post_processed_image = None  # array holding the pixel values of the post processed image
        self.original_image = None  # array holding the pixel values of the original image
        self.image_header = None  # the image header from a .FITS file
        self.input_image_dims = None  # input image dimensions in pixels of original image
        self.output_image_dims = (config['output_image_width'], config['output_image_height'])
        self.image_crop = None  # the cropped and post-processed image
        self.crop_start = None  # the pixel location in the original image of the crop start
        self.fake_im_directory = config['synthetic_image_directory']
        self.mean = None  # cropped image mean value (after 3-sigma clipping and masking)
        self.median = None  # cropped image median value (after 3-sigma clipping and masking)
        self.std = None  # cropped image standard deviation (after 3-sigma clipping and masking)
        self.stack_folder = None  # the folder path in which the cropped image will be stored
        self.configuration = config
        return

    def get_random_file(self):
        """
        The function call picks a random file (ensure only image files are contained within the directory), .fits file,
        and returns the file path is such a file is found else None
        :return: file path of randomly chosen file
        """
        # List all files and directories in the specified directory
        files = []
        for root, dirs, filenames in os.walk(self.directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        # Select a random file from the list of files
        if files:
            return random.choice(files)
        else:
            return None

    def open_image(self):
        """
        Opens the image by first using get_random_file() to obtain a file path. Splits the file into a header portion
        and an image portion. The input dimensions of the image is extracted from the header. Pre processing tasks,
        thresholding and stretching are applied to get post processed image.
        :return:
        """
        if 't' in self.configuration['options']:
            if '2' in self.configuration['options']:
                self.image_file_path = self.get_stack_file()
            else:
                self.image_file_path = os.path.join(self.configuration['test_set_directory'],
                                                self.configuration['original_image'])
        else:
            self.image_file_path = self.get_stack_file()
        if self.image_file_path:
            # open .fits file
            image_file = os.path.abspath(self.image_file_path)
            # print image info
            # fits.info(image_file)
            # get data from image (ie pixel values)
            image_data = fits.getdata(image_file, ext=0)
            header = fits.getheader(image_file, ext=0)
            self.original_image = image_data
            self.image_header = header

            self.input_image_dims = (header['NAXIS1'], header['NAXIS2'])

            self.post_processed_image = image_data
        else:
            raise ValueError("No files found in the directory.")
        return


    def get_stack_file(self):

        # Define the directory where your .fit files are located
        directory = self.configuration['stack_file_directory']

        # Get a list of all .fit files in the directory
        fit_files = [f for f in os.listdir(directory) if f.endswith('.fit')]

        # Extract the numeric part of the filenames
        file_numbers = sorted([int(f.split('.')[0]) for f in fit_files])

        # Find the maximum file number
        max_file_number = max(file_numbers)

        # Calculate the upper limit for selecting a random file
        upper_limit = max_file_number - self.configuration['num_frames']

        # Filter files that are less than or equal to the upper limit
        eligible_files = [f for f in file_numbers if f <= upper_limit]

        # Choose a random file number from the eligible files
        random_file_number = random.choice(eligible_files)

        # Construct the full filename
        random_file_name = f"{random_file_number}.fit"

        # Construct the full path to the selected file
        random_file_path = os.path.join(directory, random_file_name)

        #print(f"Random file selected: {random_file_path}")

        return random_file_path

    def open_image_known(self, path):
        """
        Opens the image by first using get_random_file() to obtain a file path. Splits the file into a header portion
        and an image portion. The input dimensions of the image is extracted from the header. Pre processing tasks,
        thresholding and stretching are applied to get post processed image.
        :return:
        """

        if path:
            # open .fits file
            image_file = os.path.abspath(path)
            # print image info
            #fits.info(image_file)
            # get data from image (ie pixel values)
            image_data = fits.getdata(image_file, ext=0)
            header = fits.getheader(image_file, ext=0)
        else:
            raise ValueError("No files found in the directory.")
        return image_data

    def process(self, image_data, row):

        # preprocessing - interval and stretching, normalize [0,1]
        interval = ManualInterval()
        stretch = AsinhStretch(a=0.99)
        min_val, max_val = interval.get_limits(image_data)
        # clip bright pixels according to percentage of max pixel value
        cut = ManualInterval(row['Stack_Mean'] - self.configuration['sigma_cutoff_bottom'] * row['Stack_Standard_Deviation'], row['Stack_Mean'] + self.configuration['sigma_cutoff_top'] * row['Stack_Standard_Deviation'])
        clipped_data = cut(image_data)
        # stretch data
        return stretch(clipped_data)

    def known_crop(self, row):
        """
        Crop an image according to output dimensions, while ensuring that the output dimensions of the image are
        satisfied. Calculate the region of allowable starting crop pixel locations, and then randomly chose from
        these locations in order to crop an image of specified dimensions. Stor in image_crop property and crop_start
        property for location.
        :return:
        """

        w2, h2 = self.output_image_dims
        path = row['Original_Image']
        image_array = self.open_image_known(path)

        w_start, h_start = row['Stack_Crop_Start']

        # Crop the image
        return image_array[h_start:h_start + h2, w_start:w_start + w2]

    def known_crop_stack(self, row, frame):
        """
        Crop an image according to output dimensions, while ensuring that the output dimensions of the image are
        satisfied. Calculate the region of allowable starting crop pixel locations, and then randomly chose from
        these locations in order to crop an image of specified dimensions. Stor in image_crop property and crop_start
        property for location.
        :return:
        """

        w2, h2 = self.output_image_dims
        path = row['Original_Image']

        # Extract the directory and filename from the original path
        directory = os.path.dirname(path)
        original_file_name = os.path.basename(path)

        # Extract the file number from the original filename
        original_number = int(original_file_name.split('.')[0])

        # Calculate the new file number
        new_number = original_number + frame

        # Construct the new filename
        new_file_name = f"{new_number}.fit"

        # Construct the full path to the new file
        new_file_path = os.path.join(directory, new_file_name)

        image_array = self.open_image_known(new_file_path)

        w_start, h_start = row['Stack_Crop_Start']

        # Crop the image
        return image_array[h_start:h_start + h2, w_start:w_start + w2]

    def random_crop(self):
        """
        Crop an image according to output dimensions, while ensuring that the output dimensions of the image are
        satisfied. Calculate the region of allowable starting crop pixel locations, and then randomly chose from
        these locations in order to crop an image of specified dimensions. Stor in image_crop property and crop_start
        property for location.
        :return:
        """
        w1, h1 = self.input_image_dims
        w2, h2 = self.output_image_dims
        if 't' in self.configuration['options'] and '2' not in self.configuration['options']:
            h_start = self.configuration['crop_start'][0]
            w_start = self.configuration['crop_start'][1]
        else:
            # Calculate the maximum valid starting pixel positions for the crop
            max_h_start = h1 - h2
            max_w_start = w1 - w2

            # Randomly choose starting pixel positions for the crop
            h_start = np.random.randint(0, max_h_start + 1)
            w_start = np.random.randint(0, max_w_start + 1)

        self.crop_start = (w_start, h_start)

        # Crop the image
        self.image_crop = self.post_processed_image[h_start:h_start + h2, w_start:w_start + w2]

        return

    def random_region_crop(self):
        """
        Crop an image according to output dimensions, while ensuring that the output dimensions of the image are
        satisfied. Calculate the region of allowable starting crop pixel locations, and then randomly chose from
        these locations in order to crop an image of specified dimensions. Stor in image_crop property and crop_start
        property for location.
        :return:
        """
        w2, h2 = self.output_image_dims
        if 't' in self.configuration['options'] and '2' not in self.configuration['options']:
            h_start = self.configuration['crop_start'][0]
            w_start = self.configuration['crop_start'][1]
        else:
            # Randomly select a region
            region = random.choice(self.configuration['regions'])
            minx = region[0]
            miny = region[1]
            maxx = region[2]
            maxy = region[3]

            # Calculate the maximum valid starting pixel positions for the crop
            max_h_start = maxy - h2
            max_w_start = maxx - w2

            # Randomly choose starting pixel positions for the crop
            h_start = np.random.randint(miny, max_h_start + 1)
            w_start = np.random.randint(minx, max_w_start + 1)

        self.crop_start = (w_start, h_start)

        # Crop the image
        self.image_crop = self.post_processed_image[h_start:h_start + h2, w_start:w_start + w2]

        return

    def view_image(self, image_data):
        """
        Given an array of pixel values, view the image. If the image has a cropped out version, add a red rectangle
        where the crop occurred in the original image.
        :param image_data:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if self.image_crop is None:  # if there is a crop of image existing
            pass
        else:
            if len(image_data) > self.output_image_dims[0]:
                # Create a rectangle patch
                rect = patches.Rectangle(xy=self.crop_start, width=self.output_image_dims[0],
                                         height=self.output_image_dims[1], linewidth=2, edgecolor='r', facecolor='none')
                # Add the rectangle patch to the axis
                ax.add_patch(rect)

        im = ax.imshow(image_data, cmap='gray')
        fig.colorbar(im)
        return

    def save_image(self):
        """
        Save the cropped out image in its own folder with a unique name of folder, but the image file is saved as
        0.jpg and 0.npy. The array saved contains normalized values 0 to 1 while the image has been rescaled from
        0 to 255.
        :return:
        """
        # Check existing stack folders and determine the next available folder name
        if 't' in self.configuration['options']:
            stacks_directory = os.path.join(self.fake_im_directory, str(self.configuration['num_stacks']), 'backgrounds', 'test')
        else:
            stacks_directory = os.path.join(self.fake_im_directory, str(self.configuration['num_stacks']), 'backgrounds', self.configuration['current_set'])

        # these two lines were to create folders of stacks, but since then have switched to saving a single array per stack for memeory concerns
        # existing_stacks = os.listdir(stacks_directory)
        # next_stack_number = len(existing_stacks)
        # Create the next available stack folder
        # next_stack_folder = os.path.join(stacks_directory, f'stack{next_stack_number}')
        # self.stack_folder = next_stack_folder
        # os.makedirs(next_stack_folder)
        # Convert normalized array to pixel values (0-255)
        # pixel_values = (self.image_crop / np.max(self.image_crop) * 255).astype(np.uint8)
        # Convert pixel values array to image
        # image = Image.fromarray(pixel_values)
        # Save the image as a JPEG file
        # image.save(os.path.join(next_stack_folder, '0.jpg'))

        # see how many stacks exist
        next_stack_number = len(
            [name for name in os.listdir(stacks_directory) if os.path.isfile(os.path.join(stacks_directory, name))])
        next_stack_file = os.path.join(stacks_directory, f'background{next_stack_number}.npy')
        self.stack_folder = next_stack_file  # this property used to be a folder when generating stacks as folders, but now is file path

        # Save the array as a NumPy file
        np.save(next_stack_file, self.image_crop)

        print(f"Created background image for stack '{next_stack_file}' and saved array.\n")

        return

    def background_stats(self):
        """
        Estimate the background statistics of the image using photoutils package. Clip the image to dampen bright
        sources. Then detect all sources above a certain limit and mask them. Then estimate the mean, median and
        standard deviation of image with masked sources.
        :return:
        """

        sigma_clip = SigmaClip(sigma=self.configuration['sigmaclip_sigma'],
                               maxiters=self.configuration['sigmaclip_maxiters'])
        threshold = detect_threshold(self.image_crop, nsigma=self.configuration['detect_threshold_nsigma'],
                                     sigma_clip=sigma_clip)
        segment_img = detect_sources(self.image_crop, threshold, npixels=self.configuration['detect_sources_npixels'])
        footprint = circular_footprint(radius=self.configuration['circular_footprint_radius'])
        mask = segment_img.make_source_mask(footprint=footprint)
        self.mean, self.median, self.std = sigma_clipped_stats(self.image_crop,
                                                               sigma=self.configuration['sigma_clipped_stats_sigma'],
                                                               mask=mask)
        return

    def stack_generator(self, num_stacks):
        """
        Main function for generating a certain number of backgrounds that are random, the number of backgrounds
        generated is specified by num_stacks parameter. First a real image is opened randomly from a specified folder.
        Next, a random crop of the original .FITS image is taken and saved in a separate folder. The backround stats
        are esimated and saved to a dataframe .csv file once all stacks are generated. The .csv file is comma separated
        with columns: columns=['Original Image', 'Saved as Stack', 'Stack Mean', 'Stack Median',
                                                'Stack Standard Deviation', 'Stack Crop Start']
        :param num_stacks: The number of random background images of specified output size to create from original input
        images.
        :return:
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        local_stats = []
        for idx in range(rank, num_stacks, size):
            print(f"Process {rank}, idx {idx}")
            self.open_image()
            self.random_region_crop()
            self.background_stats()
            local_stats.append(
                [self.image_file_path, self.mean, self.median, self.std, self.crop_start]
            )

        # Gather results from all processes
        all_stats = comm.gather(local_stats, root=0)

        if rank == 0:
            # Flatten the list of lists
            all_stats = [item for sublist in all_stats for item in sublist]
            # Create DataFrame on rank 0
            stats_df = pd.DataFrame(all_stats, columns=self.configuration['stack_file_columns'])
        else:
            # Placeholder for non-root ranks
            all_stats = None
            stats_df = None

        # Broadcast all_stats to all ranks
        all_stats = comm.bcast(all_stats, root=0)

        if all_stats is not None:
            # Recreate DataFrame on all ranks
            stats_df = pd.DataFrame(all_stats, columns=self.configuration['stack_file_columns'])

        comm.Barrier()

        # Return DataFrame for all ranks
        if 't' in self.configuration['options'] and '2' not in self.configuration['options']:
            return
        else:
            return stats_df


if __name__ == '__main__':
    aigen = BackgroundGen()
    # for i in range(0, 5):
    #     aigen.open_image()
    #     aigen.random_crop()
    #     aigen.view_image(aigen.original_image)
    # aigen.view_image(aigen.post_processed_image)
    # aigen.view_image(aigen.image_crop)
    # aigen.save_image()
    # aigen.background_stats()
    aigen.stack_generator(10000)
    plt.show()

    # open a random real image (.FITS file) that will be used to generate a synthetic tracklet
    # with fits.open('synthetic_tracklets/tycho_tracker/raw/ds1/1.fit') as hdul:
    #     data = hdul[0].data
    #     header = hdul[0].header

    # print(hdul)
    # print(data)
    # print(header)
    # print(header['BSCALE'])
    # print(header['BZERO'])
    # image_file = get_pkg_data_filename('synthetic_tracklets/tycho_tracker/raw/ds1/1.fit')
    # fits.info(image_file)
    # image_data = fits.getdata(image_file, ext=0)
    # percentage = np.arange(0, 0.5, 0.05)
    # print(image_data.shape)
    # interval = ManualInterval()
    # min_val, max_val = interval.get_limits(image_data)
    # for idx, per in enumerate(percentage):
    #     cut = ManualInterval(min_val, per * max_val)
    #     new_d = cut(image_data)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     im = ax.imshow(new_d, cmap='gray')
    #     fig.colorbar(im)
    #
    # stretches = [AsinhStretch(), LinearStretch(), LogStretch(),
    #              PowerDistStretch(), SquaredStretch(), SinhStretch(), SqrtStretch()]
    #
    # cut_str = ManualInterval(min_val, 0.15 * max_val)
    # new_d_str = cut_str(image_data)
    # for idx, stretch in enumerate(stretches):
    #     new_d_stred = stretch(new_d_str)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     im = ax.imshow(new_d_stred, cmap='gray')
    #     fig.colorbar(im)
    # plt.show()
