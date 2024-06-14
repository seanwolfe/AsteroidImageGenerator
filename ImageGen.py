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
import cv2


class ImageGen:

    def __init__(self, configs):

        self.configuration = configs
        sets = ['train', 'val']
        filenames = self.file_names(configs, sets)

        # save the config options
        yaml_contents_str = yaml.dump(configs)
        with open(os.path.join(self.configuration['synthetic_image_directory'], str(configs['num_stacks']), 'configuration.txt'), 'w') as file:
            file.write(yaml_contents_str)

        # Generate test set
        if os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']), 'testset.csv'):
            from TestSetGen import TestSetGen
            self.configuration['options'] = self.configuration['options'] + 't2'
            self.tester = TestSetGen(self.configuration)
            self.tester.set_2()

        self.configuration['options'] = self.configuration['options'].replace('t2', "")
        for idx, filenames_i in enumerate(filenames):
            if 'train' in sets[idx]:
                num_stacks = int(configs['num_stacks'] * configs['train_val_ratio'])
                self.configuration['current_set'] = 'train'
            else:
                num_stacks = int(configs['num_stacks'] * (1 - configs['train_val_ratio']))
                self.configuration['current_set'] = 'val'
            ratio = configs['real_bogus_ratio']

            configs['master_file_name'] = filenames_i[0]
            configs['distribution_file_name'] = filenames_i[1]
            configs['stack_file_name'] = filenames_i[2]
            configs['snr_file_name'] = filenames_i[3]
            configs['center_file_name'] = filenames_i[4]

            dist_file_path = os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']), configs['distribution_file_name'])
            stack_file_path = os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']),
                                           configs['stack_file_name'])
            snr_file_path = os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']),
                                         configs['snr_file_name'])
            center_file_path = os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']),
                                            configs['center_file_name'])
            master_file_path = os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']),
                                            configs['master_file_name'])

            if configs['master_file_name'] and configs['distribution_file_name'] and configs['stack_file_name'] and configs['snr_file_name'] and configs['center_file_name']:
                pass
            else:
                raise ValueError("File Path should be specified, otherwise desired file name")

            if os.path.exists(master_file_path):
                pass
            else:
                if os.path.exists(dist_file_path):
                    pass
                else:
                    dis_gen = DistribuGen(fp_pre=configs['sbd_file_name'], fp_post=configs['horizons_file_name'],
                                          fp_final=os.path.join(configs['synthetic_image_directory'], str(configs['num_stacks']), configs['distribution_file_name']),
                                          cnames_ssb=configs['sbd_file_columns'],
                                          cnames_horizon=configs['horizons_file_columns'],
                                          cnames_dist=configs['distribution_file_columns'], options=configs['options'],
                                          dist_range=configs['dist_range'], observer=configs['observer'])
                    dis_gen.generate(num_stacks, ratio)
                if os.path.exists(stack_file_path):
                    pass
                else:
                    aigen = BackgroundGen(configs)
                    aigen.stack_generator(num_stacks)
                if os.path.exists(snr_file_path):
                    pass
                else:
                    siggen = SignalGen(configs)
                    siggen.gen_snr_file()


                dist_data = pd.read_csv(dist_file_path, sep=',', header=0, names=configs['distribution_file_columns'])
                stack_data = pd.read_csv(stack_file_path, sep=',', header=0, names=configs['stack_file_columns'])
                snr_data = pd.read_csv(snr_file_path, sep=',', header=0, names=configs['snr_file_columns'])

                if os.path.exists(center_file_path):
                    pass
                else:
                    dist_data = self.streak_start_calc(dist_data=dist_data)

                center_data = pd.read_csv(center_file_path, sep=',', header=0, names=configs['center_file_columns'])
                master_data = pd.concat(
                    [dist_data.reset_index(drop=True), snr_data.reset_index(drop=True), stack_data.reset_index(drop=True),
                     center_data.reset_index(drop=True)], axis=1)

                master_data.to_csv(master_file_path, sep=',', header=True, index=False)

            self.background = BackgroundGen(configs)  # just to use some processing functions later
        return

    def streak_start_calc(self, dist_data=None, master=None):
        max_omega = self.configuration['output_image_width'] / (self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                                                                        3600 * self.configuration['pixel_scale']))
        if 't2' in self.configuration['options']:
            master.loc[master['omega'] > max_omega * 3600, 'omega'] = max_omega * 3600
            lengths = master['omega'] * self.configuration['num_frames'] * (
                    self.configuration['dt'] + self.configuration['slew_time']) / (
                              3600 * self.configuration['pixel_scale'])

            centers_x = []
            centers_y = []
            for idx, length in enumerate(lengths):
                if master['Asteroid Present'].iloc[idx] == False:
                    center_x = 0
                    center_y = 0
                else:
                    center_x = self.configuration['output_image_width'] + max(lengths) + 100
                    center_y = self.configuration['output_image_height'] + max(lengths) + 100
                    trys = 0
                    while (center_x + length * np.cos(np.deg2rad(master['Theta'].iloc[idx])) > self.configuration[
                        'output_image_width']) and (
                            center_y + length * np.sin(np.deg2rad(master['Theta'].iloc[idx])) > self.configuration[
                        'output_image_height']) and (trys < self.configuration['number_of_trys']):
                        trys += 1
                        try:
                            center_x = np.random.randint(length, self.configuration['output_image_width'] - length)
                            center_y = np.random.randint(length, self.configuration['output_image_height'] - length)
                        except ValueError:
                            print("Asteroid center id:" + str(idx))
                            if master['Theta'].iloc[idx] > 90:
                                if master['Theta'].iloc[idx] > 180:
                                    if master['Theta'].iloc[idx] > 270:
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
                                    center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8,
                                                                 self.configuration['output_image_width'])
                                    center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)
                            else:
                                center_x = np.random.randint(0, self.configuration['output_image_width'] / 8)
                                center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)

                centers_x.append(center_x)
                centers_y.append(center_y)

            master['Center x'] = centers_x
            master['Center y'] = centers_y
            return master
        else:
            dist_file_path = os.path.join(self.configuration['synthetic_image_directory'], str(self.configuration['num_stacks']), self.configuration['distribution_file_name'])

            dist_data.loc[dist_data['omega'] > max_omega * 3600, 'omega'] = max_omega * 3600
            dist_data.to_csv(dist_file_path, sep=',', header=True, index=False)

            lengths = dist_data['omega'] * self.configuration['num_frames'] * (
                    self.configuration['dt'] + self.configuration['slew_time']) / (
                              3600 * self.configuration['pixel_scale'])

            centers_x = []
            centers_y = []
            for idx, length in enumerate(lengths):
                if dist_data['Asteroid Present'].iloc[idx] == False:
                    center_x = 0
                    center_y = 0
                else:
                    center_x = self.configuration['output_image_width'] + max(lengths) + 100
                    center_y = self.configuration['output_image_height'] + max(lengths) + 100
                    trys = 0
                    while (center_x + length * np.cos(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration['output_image_width']) and (center_y + length * np.sin(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration['output_image_height']) and (trys < self.configuration['number_of_trys']):
                        trys += 1
                        try:
                            center_x = np.random.randint(length, self.configuration['output_image_width'] - length)
                            center_y = np.random.randint(length, self.configuration['output_image_height'] - length)
                        except ValueError:
                            print("Asteroid center id:" + str(idx))
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
            final_df.to_csv(os.path.join(self.configuration['synthetic_image_directory'], str(self.configuration['num_stacks']), self.configuration['center_file_name']), sep=',', header=True, index=False)
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

        for kdx, set in enumerate(['train', 'test', 'val']):
            print('Generating ' + set + ' set...' )
            self.configuration['options'] = self.configuration['options'].replace('t2', "")
            self.configuration['current_set'] = set
            if 'test' in set:
                self.configuration['options'] = self.configuration['options'] + 't2'
                master_file_path = os.path.join(self.configuration['synthetic_image_directory'],
                                                str(self.configuration['num_stacks']), 'testset.csv')
                self.master = pd.read_csv(master_file_path, sep=',', header=0,
                                          names=self.configuration['master_file_columns'])
            else:
                master_file_path = os.path.join(self.configuration['synthetic_image_directory'],
                                                str(self.configuration['num_stacks']), set + '_sample_master_' + str(
                        self.configuration['num_stacks']) + '.csv')
                self.master = pd.read_csv(master_file_path, sep=',', header=0,
                                          names=self.configuration['master_file_columns'])

            for ldx, label in enumerate(self.configuration['class_labels']):
                curr_class = self.configuration['classes'][ldx]
                current_master = self.master[self.master['Asteroid Present'] == label]
                print('Generating Asteroid Present: ' + str(label))
                for idx, row in current_master.iterrows():
                    print(row)
                    # generate track centers
                    centers_x, centers_y = self.track_centers(row)
                    big_l = row['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
                    signals = []
                    final_images = []
                    backgrounds = []

                    if 't' in self.configuration['options']:
                        if '2' in self.configuration['options']:
                            background_path = self.configuration['synthetic_image_directory']
                            background_file = row['Saved as Stack']
                            background_image = np.load(os.path.join(background_path, str(self.configuration['num_stacks']), 'backgrounds', set, background_file))
                        else:
                            #     open image
                            background_file = '0.npy'
                            background_path = row['Saved as Stack']
                            background_image = np.load(os.path.join(background_path, background_file))
                    else:
                        #     open image
                        background_path = self.configuration['synthetic_image_directory']
                        background_file = row['Saved as Stack']
                        background_image = np.load(os.path.join(background_path, str(self.configuration['num_stacks']), 'backgrounds', set, background_file))

                    for jdx in range(0, self.configuration['num_frames']):

                        backgrounds.append(background_image)

                        # generate noise for background image
                        noise = np.random.normal(loc=0, scale=row['Stack Standard Deviation'],
                                                 size=background_image.shape)

                        if row['Asteroid Present'] == False:
                            final_image = self.background.process(background_image + noise, row)
                        else:
                            # make meshgrid corresponding to entire image
                            x = np.linspace(0, self.configuration['output_image_width'] - 1,
                                            self.configuration['output_image_width'])
                            y = np.linspace(0, self.configuration['output_image_height'] - 1,
                                            self.configuration['output_image_height'])
                            big_x, big_y = np.meshgrid(x, y)

                            # calculate Gaussian at each pixel in image apply Jedicke
                            signal = self.gaussian_streak(big_x, big_y, row, centers_x[jdx], centers_y[jdx], big_l)
                            print(np.max(signal))
                            signals.append(signal)

                            if np.max(signal) > np.max(background_image):
                                signal = signal * np.max(background_image) / np.max(signal)

                            noise_mat = np.random.normal(loc=0, scale=1, size=signal.shape)
                            # final_image = self.background.process(signal + background_image + np.sqrt(signal + background_image) * noise_mat, row)
                            final_image = self.background.process(
                                signal + background_image + noise, row)

                        final_images.append(final_image)

                        # previously, save every array and image individually, but for memory concerns, save just an overall concatenated array
                        # save image
                        # Convert normalized array to pixel values (0-255)
                        # pixel_values = (final_image / np.max(final_image) * 255).astype(np.uint8)
                        # Convert pixel values array to image
                        # image = Image.fromarray(pixel_values)
                        # if 't' in self.configuration['options']:
                            # Save the array as a NumPy file
                            # np.save(os.path.join(background_path, '0{0}.npy'.format(jdx)), final_image)

                            # Save the image as a JPEG file
                            # image.save(os.path.join(background_path, '0{0}'.format(jdx)) + '.jpg')
                        # else:
                            # Save the array as a NumPy file
                            # np.save(os.path.join(background_path, '{0}{1}.npy'.format(row['Saved as Stack'], jdx)), final_image)

                            # Save the image as a JPEG file
                            # image.save(os.path.join(background_path, '{0}{1}'.format(row['Saved as Stack'], jdx)) + '.jpg')

                        # view image and background
                        if 'v' in self.configuration['options']:
                            pass
                        else:
                            if jdx > -1:
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
                                # Remove both major and minor tick labels
                                ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                                               left=False, right=False, labelbottom=False, labelleft=False)
                                # ax3.scatter(centers_x, centers_y, s=1)
                                # fig3.colorbar(im3)

                                fig4 = plt.figure()
                                ax4 = fig4.add_subplot(1, 1, 1)
                                im4 = ax4.imshow(signal, cmap='gray')
                                # ax3.set_title('Summed Signal Image 0 to 1')
                                ax4.add_patch(circle4)
                                ax4.tick_params(axis='both', which='both', bottom=False, top=False,
                                                left=False, right=False, labelbottom=False, labelleft=False)
                                # ax3.scatter(centers_x, centers_y, s=1)
                                # fig4.colorbar(im4)

                                # fig5 = plt.figure()
                                # ax5 = fig5.add_subplot(1, 1, 1)
                                # im5 = ax5.imshow(background_image, cmap='gray')
                                # ax3.set_title('Summed Signal Image 0 to 1')
                                # ax5.add_patch(circle5)
                                # ax3.scatter(centers_x, centers_y, s=1)
                                # fig5.colorbar(im5)

                                plt.show()
                            # Add a red circle with a red border (no fill)
                            # circle = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                            #                         linewidth=1)
                            # circle1 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                            #                          linewidth=1)
                            # circle2 = patches.Circle((centers_x[0], centers_y[0]), 10, edgecolor='red', facecolor='none',
                            #                          linewidth=1)

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

                    scaled_image_array = (np.array(final_images) * 255).astype(np.uint8)
                    final_image_array = np.repeat(scaled_image_array[:, :, :, np.newaxis], 3, axis=3)
                    if '2' in self.configuration['options']:
                        np.save(os.path.join(background_path, str(self.configuration['num_stacks']), 'stacks', set, curr_class, 'stack{0}.npy'.format(idx)), final_image_array)
                        self.video_file(final_image_array, os.path.join(background_path, str(self.configuration['num_stacks']), 'vids', set, curr_class, 'stack{0}'.format(idx)))
                    else:
                        np.save(os.path.join(background_path, str(self.configuration['num_stacks']), 'stacks', set, curr_class,
                                             'stack{0}.npy'.format(idx)), final_image_array)
                        self.video_file(final_image_array,
                                        os.path.join(background_path, str(self.configuration['num_stacks']), 'vids', set, curr_class,
                                                     'stack{0}'.format(idx)))

                    # print row
        return

    def gaussian_streak(self, x, y, row, x_0, y_0, big_l):
        if big_l == 0:
            big_l = 0.000001
        arg_1 = (-(x - x_0) * np.sin(np.deg2rad(row['Theta'])) + (y - y_0) * np.cos(np.deg2rad(row['Theta']))) ** 2 / (
                2 * row['Sigma_g'] ** 2)
        arg_2 = ((x - x_0) * np.cos(np.deg2rad(row['Theta'])) + (y - y_0) * np.sin(
            np.deg2rad(row['Theta'])) + big_l / 2) / (np.sqrt(2) * row['Sigma_g'])
        arg_3 = ((x - x_0) * np.cos(np.deg2rad(row['Theta'])) + (y - y_0) * np.sin(
            np.deg2rad(row['Theta'])) - big_l / 2) / (np.sqrt(2) * row['Sigma_g'])
        arg_4 = row['Expected Signal'] / (big_l * 2 * row['Sigma_g'] * np.sqrt(2 * np.pi))
        # signal
        s_xy = arg_4 * np.exp(-arg_1) * (scipy.special.erf(arg_2) - scipy.special.erf(arg_3))
        # statistical noise
        # Generate a vector with values sampled from a normal distribution
        # with mean 0 and unit variance
        # new_vector = np.random.normal(loc=0, scale=1, size=s_xy.shape)
        return s_xy #+ new_vector * np.sqrt(s_xy)

    @staticmethod
    def file_names(configuration, sets):

        num = configuration['num_stacks']
        filenames = []
        for idx, set in enumerate(sets):
            master_file_name = set + '_' + configuration['master_file_name'] + str(num) + '.csv'
            dist_file_name = set + '_' + configuration['distribution_file_name'] + str(num) + '.csv'
            stack_file_name = set + '_' +configuration['stack_file_name'] + str(num) + '.csv'
            snr_file_name = set + '_' + configuration['snr_file_name'] + str(num) + '.csv'
            center_file_name = set + '_' + configuration['center_file_name'] + str(num) + '.csv'
            filenames.append([master_file_name, dist_file_name, stack_file_name, snr_file_name, center_file_name])

        files = ['backgrounds', 'vids', 'stacks']
        sets = ['train', 'test', 'val']
        classes = configuration['classes']

        for idx, file in enumerate(files):
            for jdx, set in enumerate(sets):
                for kdx, class_i in enumerate(classes):
                    if not os.path.exists(os.path.join(configuration['synthetic_image_directory'], str(num))):
                        os.mkdir(os.path.join(configuration['synthetic_image_directory'], str(num)))
                    if not os.path.exists(
                            os.path.join(configuration['synthetic_image_directory'], str(num), file)):
                        os.mkdir(os.path.join(configuration['synthetic_image_directory'], str(num), file))
                    if not os.path.exists(
                            os.path.join(configuration['synthetic_image_directory'], str(num), file, set)):
                        os.mkdir(os.path.join(configuration['synthetic_image_directory'], str(num), file, set))
                    if file == 'backgrounds':
                        pass
                    else:
                        if not os.path.exists(
                            os.path.join(configuration['synthetic_image_directory'], str(num), file, set, class_i)):
                            os.mkdir(os.path.join(configuration['synthetic_image_directory'], str(num), file, set, class_i))


        return filenames


    @staticmethod
    def video_file(video_array, name):
        # Define the codec and create VideoWriter object
        num_frames, height, width, channels = video_array.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
        video_filename = name + '.mp4'
        video_out = cv2.VideoWriter(video_filename, fourcc, 2.0, (width, height))

        # Write each frame to the video
        for i in range(num_frames):
            frame = video_array[i]
            video_out.write(frame)  # Write the frame

        # Release everything when job is finished
        video_out.release()

        return


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ai_generator = ImageGen(config)
    ai_generator.gen_images_and_file()
