import os.path

import numpy as np
import pandas as pd
from itertools import product
from BackgroundGen import BackgroundGen
from SignalGen import SignalGen
import yaml
from ImageGen import ImageGen



class TestSetGen:

    def __init__(self, config):
        self.configuration = config
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
            dist_file_path = os.path.join(self.configuration['synthetic_image_directory'],
                                          str(self.configuration['num_stacks']),
                                          self.configuration['distribution_file_name'])

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
                    while (center_x + length * np.cos(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration[
                        'output_image_width']) and (
                            center_y + length * np.sin(np.deg2rad(dist_data['Theta'].iloc[idx])) > self.configuration[
                        'output_image_height']) and (trys < self.configuration['number_of_trys']):
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
                                    center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8,
                                                                 self.configuration['output_image_width'])
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
            final_df.to_csv(
                os.path.join(self.configuration['synthetic_image_directory'], str(self.configuration['num_stacks']),
                             self.configuration['center_file_name']), sep=',', header=True, index=False)
            return dist_data

    def set_2(self):

        num_samples_per_bin = self.configuration['num_samples_per_bin']
        num_bins = self.configuration['num_bins']

        min_omega = self.configuration['min_omega']
        max_omega = self.configuration['max_omega']
        omegas = np.linspace(min_omega, max_omega, num=num_bins * num_samples_per_bin)

        min_v = self.configuration['min_apparent_magnitude']
        max_v = self.configuration['max_apparent_magnitude']
        vs = np.linspace(min_v, max_v, num=num_bins * num_samples_per_bin)

        min_sigma = self.configuration['min_sigma_g']
        max_sigma = self.configuration['max_sigma_g']
        sigmas = np.linspace(min_sigma, max_sigma, num_bins * num_samples_per_bin)
        combos = np.array(list(product(omegas, vs, sigmas)))
        thetas = np.random.uniform(0, 360, len(combos))
        zero_col = np.zeros_like(thetas)
        self.background = BackgroundGen(self.configuration)
        self.signal_gen = SignalGen(self.configuration)
        stack_data = self.background.stack_generator(len(combos))
        master_array = np.array([zero_col, combos[:, 0], zero_col, zero_col, zero_col, thetas, combos[:, 2], zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col])
        snr_calc = pd.DataFrame(master_array.T, columns=self.configuration['master_file_columns'])
        snr_calc['Original Image'] = stack_data['Original Image']
        snr_calc['Saved as Stack'] = stack_data['Saved as Stack']
        snr_calc['Stack Mean'] = stack_data['Stack Mean']
        snr_calc['Stack Median'] = stack_data['Stack Median']
        snr_calc['Stack Standard Deviation'] = stack_data['Stack Standard Deviation']
        snr_calc['Stack Crop Start'] = stack_data['Stack Crop Start']
        snr_calc['Asteroid Present'] = True
        snr_calc['H'] = combos[:, 1]
        testset_1 = self.signal_gen.gen_snr_file(master=snr_calc, v_s_s=combos[:, 1])
        new_master = self.streak_start_calc(master=testset_1)
        new_master.to_csv(os.path.join(self.configuration['synthetic_image_directory'], str(self.configuration['num_stacks']), 'testset.csv'), sep=',', index=False,
                            header=True)
        return




    def set_1(self):
        # Theta, Sigma_g, Expected SNR, Original Image, Crop start, Center x, Center y
        self.configuration['num_stacks'] = 1
        self.background = BackgroundGen(self.configuration)
        self.signal_gen = SignalGen(self.configuration)
        self.background.stack_generator(self.configuration['num_stacks'])
        signal = [0]
        combos1 = list(product(self.configuration['h'], self.configuration['omega'], self.configuration['r_oa'],
                               self.configuration['r_sa'], self.configuration['alpha'], self.configuration['theta'],
                               [1], self.configuration['g_12'],
                               [0], signal, [self.configuration['original_image']],
                               [self.background.stack_folder], [self.background.mean], [self.background.median],
                               [self.background.std],
                               [self.configuration['crop_start']], [self.configuration['center_x']],
                               [self.configuration['center_y']]))
        combos2 = list(product([0], [1000], [0],
                               [0], self.configuration['alpha'], self.configuration['theta'],
                               self.configuration['sigma'], self.configuration['g_12'],
                               self.configuration['expected_snr'], signal, [self.configuration['original_image']],
                               [self.background.stack_folder], [self.background.mean], [self.background.median],
                               [self.background.std],
                               [self.configuration['crop_start']], [self.configuration['center_x']],
                               [self.configuration['center_y']]))
        snr_calc = pd.DataFrame(combos1, columns=self.configuration['master_file_columns'])
        testset_1 = self.signal_gen.gen_snr_file(snr_calc)
        signal_calc = pd.DataFrame(combos2, columns=self.configuration['master_file_columns'])
        signal_calc['Expected Signal'] = self.signal_gen.signal_calc_test(signal_calc)
        self.testset = pd.concat([testset_1, signal_calc])
        self.testset.reset_index(drop=True, inplace=True)
        self.testset.to_csv(os.path.join(self.configuration['test_set_directory'], 'testset.csv'), sep=',', index=False,
                            header=True)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    tester = TestSetGen(config)
    tester.set_2()
