import pandas as pd
from astroquery.jplhorizons import Horizons
import re
from astropy.time import Time
import astropy.units as u
import numpy as np
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import matplotlib.pyplot as plt
import scipy.stats as sp


class DistribuGen:
    # properties

    def __init__(self, fp_pre=None, cnames_ssb=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y',
                                                'data_arc', 'n_obs_used', 'H', 'first_obs', 'epoch'],
                 cnames_horizon=['RA_rate', 'DEC_rate', 'x', 'y', 'z', 'alpha'],
                 cnames_dist=['H', 'omega', 'r_as', 'r_ao', 'alpha'], options='s'):
        self.file_path_pre = fp_pre
        self.ssb_names = cnames_ssb
        self.h_names = cnames_horizon
        self.dist_names = cnames_dist
        self.file_path_post = None
        self.ssb_data = None
        self.h_data = None
        self.dist_list = None
        self.options = options
        self.samples = None
        return


    def database_parser_ssb(self):
        self.ssb_data = pd.read_csv(self.file_path_pre, sep=',', header=0,
                                    names=self.ssb_names)
        return

    def database_parser_h(self):
        self.h_data = pd.read_csv(self.file_path_post, sep=',', header=0,
                                  names=self.h_names)
        return


    def jpl_querier(self, row):
        # Integration step (in days)
        int_step_jpl = 1
        int_step_jpl_unit = 'd'

        # Observation locations
        earth_obs = '500'  # Geocentric earth
        # earth_obs = '675'  # Palomar
        sun_obs = '500@10'
        # emb_obs = '500@3'  # Earth-moon barycenter (work only for jpl)

        # extract relevant parameters
        nea_name = self.parse_name(row['full_name'])
        first_obs = Time(row['first_obs'], format='isot', scale='tai')
        end_obs = first_obs + 1 * u.d
        start_time = first_obs.to_value('isot', 'date')
        end_time = end_obs.to_value('isot', 'date')

        ###############################################
        # Nea data with respect to Earth observer
        ###############################################

        nea = Horizons(id=nea_name, location=earth_obs,
                       epochs={'start': start_time, 'stop': end_time,
                               'step': str(int_step_jpl) + int_step_jpl_unit})

        nea_eph = nea.ephemerides()
        ra_rate = nea_eph[0]['RA_rate']  # in arcsec/hour
        dec_rate = nea_eph[0]['DEC_rate']  # in arcsec/hour
        nea_state = nea.vectors()
        nea_geo_x = nea_state[0]['x']  # AU
        nea_geo_y = nea_state[0]['y']
        nea_geo_z = nea_state[0]['z']

        ###############################################
        # Observer data
        ##############################################

        nea_helio = Horizons(id=nea_name, location=sun_obs,
                             epochs={'start': start_time, 'stop': end_time,
                                     'step': str(int_step_jpl) + int_step_jpl_unit})

        nea_helio_state = nea_helio.vectors()
        nea_helio_x = nea_helio_state[0]['x']
        nea_helio_y = nea_helio_state[0]['y']
        nea_helio_z = nea_helio_state[0]['z']

        alpha = nea_eph[0]['alpha']  # in degrees
        omega = np.sqrt(ra_rate ** 2 + dec_rate ** 2)  # arcsec/hour
        r_oa = np.linalg.norm([nea_geo_x, nea_geo_y, nea_geo_z])  # au
        r_as = np.linalg.norm([nea_helio_x, nea_helio_y, nea_helio_z])  # au

        return omega, r_oa, r_as, alpha

    def fit_distribution(self):

        summaries = []
        paramss = []

        # get dataset
        self.database_parser_h()
        data = self.h_data
        col_names = self.dist_names

        # Filter out non-finite data
        for idx, col_name in enumerate(col_names):

            finite_values = ~data[col_name].isna() & data[col_name].apply(np.isfinite)
            # Filter the DataFrame to show rows with non-finite values
            rows_with_non_finite = data[finite_values]
            d = rows_with_non_finite[col_name].values
            if 'q' in self.options:
                f = Fitter(d, distributions=get_common_distributions())
            else:
                f = Fitter(d)
            f.fit()
            if 'v' in self.options:
                summary = f.summary(plot=False)
            else:
                fig = plt.figure()
                summary = f.summary()
                plt.xlabel(col_name)
                plt.ylabel('Frequency')
            summaries.append(summary)
            params = f.fitted_param[summary.index[0]]
            paramss.append(params)

        dist_list = []
        for idx, summarie in enumerate(summaries):
            params = paramss[idx]
            best_dist = summarie.index[0]
            print("The best fitting distribution for {0} is the {1} distribution with parameters: {2}".format(
                col_names[idx], best_dist, params))
            print("\nA summary of the fitting:\n")
            print(summarie['sumsquare_error'])
            print("\n")

            dist_list.append([col_names[idx], best_dist, params])

        self.dist_list = dist_list

        return

    def sample_gen(self, num_samples):

        # self.samples = pd.DataFrame(columns=self.dist_names)
        pre_append = []
        for idx, dist in enumerate(self.dist_list):
            # unpack
            # col_name = dist[0]
            best_dist = dist[1]
            params = dist[2]

            # Check if the distribution name is valid
            if hasattr(sp, best_dist):
                # Get the distribution object using getattr()
                dist = getattr(sp, best_dist)
                # Generate a random variable from the distribution
                random_variable = dist.rvs(*params, size=num_samples)
                pre_append.append(random_variable)

            else:
                print("Invalid distribution name")
                return None

        self.samples = pd.DataFrame(np.array(pre_append).reshape(num_samples, len(self.dist_names)),
                                    columns=self.dist_names)
        return

    @staticmethod
    def parse_name(full_name):
        matches = re.findall(r"\((.*?)\)", full_name)
        return matches[0]


    def generate(self, num_samples=1):

        sns.set(rc={'figure.figsize': (7, 6)})
        sns.set_style('ticks')
        sns.set_context("paper", font_scale=2)
        self.database_parser_ssb()
        self.fit_distribution()
        if 'm' in self.options:
            pass
        else:
            self.sample_gen(num_samples)

        if 'v' in self.options:
            pass
        else:
            plt.show()

        return


if __name__ == '__main__':
    # two functions so far
add user-friendly
    # optional query SSB database

    # first is to query JPL to get relevant quantities

    # second is to fit the distribution for each parameter
    # this should return each characteristic histogram with top-5 plotted
    # this should return top-5 sum-of-square errors
    # this should return the best distribtution name, parameters, and sample generators
    # options - 'v': do not include visualization
    #           'q': quick fit of distribution looking at only common ones
    #           's': standard full package (default)
    #           'm': do not generate samples
    dis_gen = DistribuGen(fp_pre='NEA_database.csv',
                          cnames_horizon=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y',
                                          'data_arc', 'n_obs_used', 'H', 'first_obs', 'epoch'],
                          cnames_dist=['H', 'i'], options='q')
    dis_gen.file_path_post = 'NEA_database.csv'
    number_of_samples = 1000000
    dis_gen.generate(number_of_samples)

