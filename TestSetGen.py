import os.path

import numpy as np
import pandas as pd
from itertools import product
from BackgroundGen import BackgroundGen
from SignalGen import SignalGen
import yaml


class TestSetGen:

    def __init__(self, config):
        # Theta, Sigma_g, Expected SNR, Original Image, Crop start, Center x, Center y
        self.configuration = config
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

        return


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    tester = TestSetGen(config)
