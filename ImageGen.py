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

plt.style.use(astropy_mpl_style)


class ImageGen:

    # properties

    def __init__(self, real_image_dir='synthetic_tracklets/real_image_stacks',
                 synthetic_image_dir='synthetic_tracklets/synthetic_image_stacks', interval_threshold=0.15, stretch=0.1,
                 output_image_width=224, ouput_image_height=224):
        self.directory = real_image_dir
        self.image_file_path = None
        self.interval_thresh = interval_threshold
        self.stretch_factor = stretch
        self.post_processed_image = None
        self.original_image = None
        self.image_header = None
        self.input_image_dims = None
        self.output_image_dims = (output_image_width, ouput_image_height)
        self.image_crop = None
        self.crop_start = None
        self.fake_im_directory = synthetic_image_dir
        self.mean = None
        self.median = None
        self.std = None
        self.stack_folder = None
        return

    def get_random_file(self):
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
        self.image_file_path = self.get_random_file()
        if self.image_file_path:
            # open .fits file
            image_file = get_pkg_data_filename(self.image_file_path)
            # print image info
            fits.info(image_file)
            # get data from image (ie pixel values)
            image_data = fits.getdata(image_file, ext=0)
            header = fits.getheader(image_file, ext=0)
            self.original_image = image_data
            self.image_header = header

            self.input_image_dims = (header['NAXIS1'], header['NAXIS2'])

            # preprocessing - interval and stretching, normalize [0,1]
            interval = ManualInterval()
            stretch = AsinhStretch(a=self.stretch_factor)
            min_val, max_val = interval.get_limits(image_data)
            # clip bright pixels according to percentage of max pixel value
            cut = ManualInterval(min_val, self.interval_thresh * max_val)
            clipped_data = cut(image_data)
            # stretch data
            self.post_processed_image = stretch(clipped_data)

        else:
            print("No files found in the directory.")
        return

    def random_crop(self):
        w1, h1 = self.input_image_dims
        w2, h2 = self.output_image_dims
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

    def view_image(self, image_data):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if self.image_crop is None:
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
        # Check existing stack folders and determine the next available folder name
        stacks_directory = self.fake_im_directory
        existing_stacks = os.listdir(stacks_directory)
        next_stack_number = len(existing_stacks)

        # Create the next available stack folder
        next_stack_folder = os.path.join(stacks_directory, f'stack{next_stack_number}')
        self.stack_folder = next_stack_folder
        os.makedirs(next_stack_folder)

        # Convert normalized array to pixel values (0-255)
        pixel_values = (self.image_crop * 255).astype(np.uint8)

        # Convert pixel values array to image
        image = Image.fromarray(pixel_values)

        # Save the array as a NumPy file
        np.save(os.path.join(next_stack_folder, '0.npy'), self.image_crop)

        # Save the image as a JPEG file
        image.save(os.path.join(next_stack_folder, '0.jpg'))

        print(f"Created stack folder '{next_stack_folder}' and saved array and image.\n")

        return

    def background_stats(self):
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        threshold = detect_threshold(self.image_crop, nsigma=2.0, sigma_clip=sigma_clip)
        segment_img = detect_sources(self.image_crop, threshold, npixels=10)
        footprint = circular_footprint(radius=10)
        mask = segment_img.make_source_mask(footprint=footprint)
        self.mean, self.median, self.std = sigma_clipped_stats(self.image_crop, sigma=3.0, mask=mask)
        return

    def stack_generator(self, num_stacks):
        stats = []
        for idx in range(0, num_stacks):
            self.open_image()
            self.random_crop()
            self.save_image()
            self.background_stats()
            stats.append(
                [os.path.basename(self.image_file_path), os.path.basename(self.stack_folder), self.mean, self.median,
                 self.std, self.crop_start])

        stats_df = pd.DataFrame(stats, columns=['Original Image', 'Saved as Stack', 'Stack Mean', 'Stack Median',
                                                'Stack Standard Deviation', 'Stack Crop Start'])
        stats_df.to_csv(path_or_buf='stack_stats.csv', sep=',', header=True, index=False)
        return


if __name__ == '__main__':
    aigen = ImageGen()
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
