import os.path
import pandas as pd
from DistribuGen import DistribuGen
from BackgroundGen import BackgroundGen
from SignalGen import SignalGen
import yaml
import numpy as np


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
                self.streak_start_calc(dist_data)

            center_data = pd.read_csv(center_file, sep=',', header=0, names=configs['center_file_columns'])
            master_data = pd.concat(
                [dist_data.reset_index(drop=True), snr_data.reset_index(drop=True), stack_data.reset_index(drop=True),
                 center_data.reset_index(drop=True)], axis=1)
            master_data.to_csv(master_file, sep=',', header=True, index=False)
        self.master = pd.read_csv(master_file, sep=',', header=0, names=configs['master_file_columns'])
        return

    def streak_start_calc(self, dist_data):
        lengths = 2 * dist_data['omega'] * self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                          3600 * self.configuration['pixel_scale'])
        centers_x = pd.DataFrame(lengths).applymap(
            lambda x: np.random.randint(x, self.configuration['output_image_width'] - x))
        centers_y = pd.DataFrame(lengths).applymap(
            lambda y: np.random.randint(y, self.configuration['output_image_height'] - y))

        centers_x.rename(columns={'omega': 'Center x'}, inplace=True)
        centers_y.rename(columns={'omega': 'Center y'}, inplace=True)
        centers_x.reset_index(drop=True)
        centers_y.reset_index(drop=True)
        final_df = pd.concat([centers_x, centers_y], axis=1)
        final_df.to_csv(config['center_file_name'], sep=',', header=True, index=False)
        return

    def track_centers(self, row):

        center_x0 = row['Center x']
        center_y0 = row['Center y']
        centers_x = []
        centers_y = []
        for kdx in range(0, self.configuration['num_frames']):
            center_xp1 = center_x0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                        self.configuration['dt'] + self.configuration['slew_time'] * np.cos(np.deg2rad(row['Theta'])))
            center_yp1 = center_y0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                    self.configuration['dt'] + self.configuration['slew_time'] * np.sin(np.deg2rad(row['Theta'])))
            centers_x.append(center_xp1)
            centers_y.append(center_yp1)
        return centers_x, centers_y

    def gen_images_and_file(self):

        for idx, row in self.master.iterrows():

            # generate track centers
            centers_x, centers_y = self.track_centers(row)

            # generate signals for each track center
            length = row['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
            crop_dist = int(length / 2 + self.configuration['signal_crop_threshold'] * row['Sigma_g'])

            for jdx in range(0, self.configuration['num_frames']):
                # open image

                # make meshgrid corresponding to entire image

                # calculate Gaussian at each pixel in image

                # add meshgrid to image

                # center_x = centers_x[jdx]
                # center_y = centers_y[jdx]
                # x = np.linspace(center_x - crop_dist, center_x + crop_dist, 2 * crop_dist)
                # y = np.linspace(center_y - crop_dist, center_y + crop_dist, 2 * crop_dist)


                # add signal to background

                # save image
        return

    @staticmethod
    def file_names(configuration):
        num = configuration['num_stacks']
        master_file_name = config['master_file_name'] + str(num) + '.csv'
        dist_file_name = config['distribution_file_name'] + str(num) + '.csv'
        stack_file_name = config['stack_file_name'] + str(num) + '.csv'
        snr_file_name = config['snr_file_name'] + str(num) + '.csv'
        center_file_name = config['center_file_name'] + str(num) + '.csv'
        return master_file_name, dist_file_name, stack_file_name, snr_file_name, center_file_name


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ai_generator = ImageGen(config)
    ai_generator.gen_images_and_file()
