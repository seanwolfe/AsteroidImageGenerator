import os.path
import pandas as pd
from DistribuGen import DistribuGen
from BackgroundGen import BackgroundGen
from SignalGen import SignalGen
import yaml
import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from TestSetGen import TestSetGen


class ImageGen:

    def __init__(self, configs):

        num_stacks = configs['num_stacks']
        master_file, dist_file, stack_file, snr_file, center_file = self.file_names(configs)
        configs['master_file_name'] = master_file
        configs['distribution_file_name'] = dist_file
        configs['stack_file_name'] = stack_file
        configs['snr_file_name'] = snr_file
        configs['center_file_name'] = center_file
        self.configuration = configs
        if 't' in configs['options']:
            self.tester = TestSetGen(self.configuration)
            self.master = self.tester.testset
        else:
            if master_file and dist_file and stack_file and snr_file and center_file:
                pass
            else:
                raise ValueError("File Path should be specified, otherwise desired file name")

            if os.path.exists(master_file):
                pass
            else:
                if os.path.exists(dist_file):
                    pass
                else:
                    dis_gen = DistribuGen(fp_pre=configs['sbd_file_name'], fp_post=configs['horizons_file_name'],
                                          fp_final=dist_file,
                                          cnames_ssb=configs['sbd_file_columns'],
                                          cnames_horizon=configs['horizons_file_columns'],
                                          cnames_dist=configs['distribution_file_columns'], options=configs['options'],
                                          dist_range=configs['dist_range'], observer=configs['observer'])
                    dis_gen.generate(num_stacks)
                if os.path.exists(stack_file):
                    pass
                else:
                    aigen = BackgroundGen(configs)
                    aigen.stack_generator(num_stacks)
                if os.path.exists(snr_file):
                    pass
                else:
                    siggen = SignalGen(configs)
                    siggen.gen_snr_file()

                dist_data = pd.read_csv(dist_file, sep=',', header=0, names=configs['distribution_file_columns'])
                stack_data = pd.read_csv(stack_file, sep=',', header=0, names=configs['stack_file_columns'])
                snr_data = pd.read_csv(snr_file, sep=',', header=0, names=configs['snr_file_columns'])

                if os.path.exists(center_file):
                    pass
                else:
                    dist_data = self.streak_start_calc(dist_data)

                center_data = pd.read_csv(center_file, sep=',', header=0, names=configs['center_file_columns'])
                master_data = pd.concat(
                    [dist_data.reset_index(drop=True), snr_data.reset_index(drop=True), stack_data.reset_index(drop=True),
                     center_data.reset_index(drop=True)], axis=1)
                master_data.to_csv(master_file, sep=',', header=True, index=False)
            self.master = pd.read_csv(master_file, sep=',', header=0, names=configs['master_file_columns'])
        return

    def streak_start_calc(self, dist_data):

        max_omega = self.configuration['output_image_width'] / (self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                                                                        3600 * self.configuration['pixel_scale']))
        dist_data.loc[dist_data['omega'] > max_omega * 3600, 'omega'] = max_omega * 3600
        dist_data.to_csv(self.configuration['distribution_file_name'], sep=',', header=True, index=False)

        lengths = dist_data['omega'] * self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                          3600 * self.configuration['pixel_scale'])

        centers_x = []
        centers_y = []
        for idx, length in enumerate(lengths):
            center_x = self.configuration['output_image_width'] + max(lengths) + 100
            center_y = self.configuration['output_image_height'] + max(lengths) + 100
            while (center_x + length * np.cos(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration['output_image_width']) and (center_y + length * np.sin(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration['output_image_height']):
                try:
                    center_x = np.random.randint(length, self.configuration['output_image_width'] - length)
                    center_y = np.random.randint(length, self.configuration['output_image_height'] - length)
                except ValueError:
                    print(idx)
                    print(dist_data['Theta'].iloc[idx])
                    if dist_data['Theta'].iloc[idx] > 90:
                        if dist_data['Theta'].iloc[idx] > 180:
                            if dist_data['Theta'].iloc[idx] > 270:
                                center_x = np.random.randint(0, self.configuration[
                                    'output_image_width'] / 8)
                                center_y = np.random.randint(7 * self.configuration['output_image_height'] / 8,
                                                             self.configuration['output_image_height'])
                            else:
                                center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8,
                                                             self.configuration['output_image_width'])
                                center_y = np.random.randint(7 * self.configuration['output_image_height'] / 8,
                                                             self.configuration['output_image_height'])
                        else:
                            center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8, self.configuration['output_image_width'])
                            center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)
                    else:
                        center_x = np.random.randint(0, self.configuration['output_image_width'] / 8)
                        center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)

            centers_x.append(center_x)
            centers_y.append(center_y)

        centers_x = pd.DataFrame(centers_x, columns=['Center x'])
        centers_y = pd.DataFrame(centers_y, columns=['Center y'])
        centers_x.reset_index(drop=True)
        centers_y.reset_index(drop=True)
        final_df = pd.concat([centers_x, centers_y], axis=1)
        final_df.to_csv(self.configuration['center_file_name'], sep=',', header=True, index=False)
        return dist_data

    def track_centers(self, row):

        center_x0 = row['Center x']
        center_y0 = row['Center y']
        centers_x = []
        centers_y = []
        for kdx in range(0, self.configuration['num_frames']):
            center_xp1 = center_x0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                    self.configuration['dt'] + self.configuration['slew_time']) * np.cos(np.deg2rad(row['Theta']))
            center_yp1 = center_y0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                    self.configuration['dt'] + self.configuration['slew_time']) * np.sin(np.deg2rad(row['Theta']))
            centers_x.append(center_xp1)
            centers_y.append(center_yp1)
        return centers_x, centers_y

    def gen_images_and_file(self):

        for idx, row in self.master.iterrows():
            print(row)
            # generate track centers
            centers_x, centers_y = self.track_centers(row)
            big_l = row['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
            signals = []
            final_images = []
            backgrounds = []
            for jdx in range(0, self.configuration['num_frames']):

                if 't' in self.configuration['options']:
                    #     open image
                    background_file = '0.npy'
                    background_path = row['Saved as Stack']
                    background_image = np.load(os.path.join(background_path, background_file))
                else:
                    #     open image
                    background_path = self.configuration['synthetic_image_directory'] + '/' + row[
                        'Saved as Stack']
                    background_file = '0.npy'
                    background_image = np.load(os.path.join(background_path, background_file))

                backgrounds.append(background_image)
                # make meshgrid corresponding to entire image
                x = np.linspace(0, self.configuration['output_image_width'] - 1,
                                self.configuration['output_image_width'])
                y = np.linspace(0, self.configuration['output_image_height'] - 1,
                                self.configuration['output_image_height'])
                big_x, big_y = np.meshgrid(x, y)

                # calculate Gaussian at each pixel in image apply Jedicke
                signal = self.gaussian_streak(big_x, big_y, row, centers_x[jdx], centers_y[jdx], big_l)
                signals.append(signal)

                # add meshgrid to image
                final_image = self.tester.background.process(signal + background_image)
                final_images.append(final_image)

                # save image
                # Convert normalized array to pixel values (0-255)
                pixel_values = (final_image / np.max(final_image) * 255).astype(np.uint8)

                # Convert pixel values array to image
                image = Image.fromarray(pixel_values)

                if 't' in self.configuration['options']:
                    # Save the array as a NumPy file
                    np.save(os.path.join(background_path, '0{0}.npy'.format(jdx)), final_image)

                    # Save the image as a JPEG file
                    image.save(os.path.join(background_path, '0{0}'.format(jdx)) + '.jpg')

                else:
                    # Save the array as a NumPy file
                    np.save(os.path.join(background_path, '{0}{1}.npy'.format(row['Saved as Stack'], jdx)), final_image)

                    # Save the image as a JPEG file
                    image.save(os.path.join(background_path, '{0}{1}'.format(row['Saved as Stack'], jdx)) + '.jpg')

                # view image and background
                if 'v' in self.configuration['options']:
                    pass
                else:
                    # view cumulated image
                    circle3 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                             linewidth=1)
                    circle4 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                             linewidth=1)
                    circle5 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                             linewidth=1)

                    summed_signal = sum(signals)
                    fig3 = plt.figure()
                    ax3 = fig3.add_subplot(1, 1, 1)
                    im3 = ax3.imshow(final_image, cmap='gray')
                    # ax3.set_title('Summed Signal Image 0 to 1')
                    ax3.add_patch(circle3)
                    # ax3.scatter(centers_x, centers_y, s=1)
                    fig3.colorbar(im3)

                    fig4 = plt.figure()
                    ax4 = fig4.add_subplot(1, 1, 1)
                    im4 = ax4.imshow(signal, cmap='gray')
                    # ax3.set_title('Summed Signal Image 0 to 1')
                    ax4.add_patch(circle4)
                    # ax3.scatter(centers_x, centers_y, s=1)
                    fig4.colorbar(im4)

                    # fig5 = plt.figure()
                    # ax5 = fig5.add_subplot(1, 1, 1)
                    # im5 = ax5.imshow(background_image, cmap='gray')
                    # ax3.set_title('Summed Signal Image 0 to 1')
                    # ax5.add_patch(circle5)
                    # ax3.scatter(centers_x, centers_y, s=1)
                    # fig5.colorbar(im5)

                    plt.show()
                    # Add a red circle with a red border (no fill)
                    circle = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                            linewidth=1)
                    circle1 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                             linewidth=1)
                    circle2 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                                             linewidth=1)

                    # Add the circle to the plot
                    # fig = plt.figure()
                    # ax = fig.add_subplot(1, 1, 1)
                    # im = ax.imshow(final_image, cmap='gray')
                    # ax.set_title('Final Image 0 to 1')
                    # ax.add_patch(circle)
                    # fig.colorbar(im)
                    #
                    # fig1 = plt.figure()
                    # ax1 = fig1.add_subplot(1, 1, 1)
                    # im1 = ax1.imshow(background_image, cmap='gray')
                    # ax1.set_title('Background Image 0 to 1')
                    # ax1.add_patch(circle1)
                    # fig1.colorbar(im1)

                    # fig2 = plt.figure()
                    # ax2 = fig2.add_subplot(1, 1, 1)
                    # im2 = ax2.imshow(signal, cmap='gray')
                    # ax2.set_title('Signal Image 0 to 1')
                    # ax2.scatter(centers_x[jdx], centers_y[jdx], s=1)
                    # ax2.plot([centers_x[0], centers_x[0] + row['omega'] * self.configuration['dt'] / self.configuration['pixel_scale'] / 3600 * np.cos(np.deg2rad(row['Theta']))],
                    #          [centers_y[0], centers_y[0] + row['omega'] * self.configuration['dt'] / self.configuration['pixel_scale'] / 3600 * np.sin(np.deg2rad(row['Theta']))])
                    # ax2.add_patch(circle2)
                    # fig2.colorbar(im2)




            # print row
        return

    def gaussian_streak(self, x, y, row, x_0, y_0, big_l):
        arg_1 = (-(x - x_0) * np.sin(np.deg2rad(row['Theta'])) + (y - y_0) * np.cos(np.deg2rad(row['Theta']))) ** 2 / (
                2 * row['Sigma_g'] ** 2)
        arg_2 = ((x - x_0) * np.cos(np.deg2rad(row['Theta'])) - (y - y_0) * np.sin(
            np.deg2rad(row['Theta'])) + big_l / 2) / (np.sqrt(2) * row['Sigma_g'])
        arg_3 = ((x - x_0) * np.cos(np.deg2rad(row['Theta'])) - (y - y_0) * np.sin(
            np.deg2rad(row['Theta'])) - big_l / 2) / (np.sqrt(2) * row['Sigma_g'])
        arg_4 = row['Expected Signal'] / (big_l * 2 * row['Sigma_g'] * np.sqrt(2 * np.pi))
        arg_4_test = row['Expected Signal']

        # signal
        s_xy = arg_4_test * np.exp(-arg_1) * (scipy.special.erf(arg_2) - scipy.special.erf(arg_3))
        # background
        # background_flux = (big_l + 2 * self.configuration['num_sigmas'] * row['Sigma_g']) * (
        #             self.configuration['num_sigmas'] * row['Sigma_g']) * row['Stack Mean']
        # statistical noise
        # Generate a vector with values sampled from a normal distribution
        # with mean 0 and unit variance
        new_vector = np.random.normal(loc=0, scale=1, size=s_xy.shape)
        return s_xy + new_vector * np.sqrt(s_xy)

    @staticmethod
    def file_names(configuration):
        num = configuration['num_stacks']
        master_file_name = configuration['master_file_name'] + str(num) + '.csv'
        dist_file_name = configuration['distribution_file_name'] + str(num) + '.csv'
        stack_file_name = configuration['stack_file_name'] + str(num) + '.csv'
        snr_file_name = configuration['snr_file_name'] + str(num) + '.csv'
        center_file_name = configuration['center_file_name'] + str(num) + '.csv'
        return master_file_name, dist_file_name, stack_file_name, snr_file_name, center_file_name


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ai_generator = ImageGen(config)
    ai_generator.gen_images_and_file()
