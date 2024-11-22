import os.path
import mpi4py.rc

mpi4py.rc.threads = False
from mpi4py import MPI
import numpy as np
import pandas as pd
from itertools import product
from BackgroundGen_par_stack import BackgroundGen
from SnrGen import SnrGen
import yaml
from ImageGen_par_stack import ImageGenPar


class TestSetGen:

    def __init__(self, config):
        self.configuration = config
        return

    def set_2(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        path = os.path.join(self.configuration['test_set_directory'],
                            'test_' + str(self.configuration['min_omega']) + str(
                                self.configuration['max_omega']) + str(
                                self.configuration['min_apparent_magnitude']) + str(
                                self.configuration['max_apparent_magnitude']))
        self.configuration['test_set_directory'] = path
        directory = os.path.join(self.configuration['test_set_directory'], 'vids')

        if rank == 0:
            os.mkdir(path)
            os.mkdir(directory)
        else:
            pass

        comm.Barrier()

        num_samples_per_bin = self.configuration['num_samples_per_bin']
        num_bins = self.configuration['num_bins']

        min_omega = self.configuration['min_omega']
        max_omega = self.configuration['max_omega']
        omegas = np.linspace(min_omega, max_omega, num=num_bins * num_samples_per_bin)

        min_v = self.configuration['min_apparent_magnitude']
        max_v = self.configuration['max_apparent_magnitude']
        vs = np.linspace(min_v, max_v, num=num_bins * num_samples_per_bin)

        combos = np.array(list(product(omegas, vs)))

        min_sigma = self.configuration['min_sigma_g']
        max_sigma = self.configuration['max_sigma_g']
        sigmas = np.random.uniform(min_sigma, max_sigma, len(combos))
        thetas = np.random.uniform(0, 360, len(combos))
        zero_col = np.zeros_like(thetas)

        self.configuration['num_stacks'] = len(combos)
        self.configuration['real_bogus_ratio'] = 1
        self.background = BackgroundGen(self.configuration)
        stack_data = self.background.stack_generator(self.configuration['num_stacks'])

        master_array = np.array(
            [zero_col, combos[:, 0], zero_col, zero_col, zero_col, thetas, sigmas, zero_col, zero_col, zero_col,
             zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col,
             zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col, zero_col,
             zero_col, zero_col, zero_col])

        snr_calc = pd.DataFrame(master_array.T, columns=self.configuration['master_file_columns'])
        snr_calc.loc[:, self.configuration['stack_file_columns']] = stack_data
        snr_calc['Asteroid_Present'] = True
        snr_calc['H'] = combos[:, 1]

        # SNR
        self.signal_gen = SnrGen(self.configuration, snr_calc.loc[:, self.configuration['distribution_file_columns']])
        testset_1 = self.signal_gen.gen_snr_file(master=snr_calc, v_s_s=combos[:, 1])

        imager = ImageGenPar(self.configuration)
        new_master = imager.streak_start_calc(testset_1)
        if rank == 0:
            new_master.to_csv(
                os.path.join(self.configuration['test_set_directory'], self.configuration['test_set_master']), sep=',',
                index=False,
                header=True)

        comm.Barrier()

        imager.gen_images_and_file()
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
        signal_calc['Expected_Signal'] = self.signal_gen.signal_calc_test(signal_calc)
        self.testset = pd.concat([testset_1, signal_calc])
        self.testset.reset_index(drop=True, inplace=True)
        self.testset.to_csv(os.path.join(self.configuration['test_set_directory'], 'testset.csv'), sep=',', index=False,
                            header=True)


if __name__ == '__main__':
    with open('config_par_local.yaml', 'r') as f:
        config = yaml.safe_load(f)
    tester = TestSetGen(config)
    tester.set_2()
