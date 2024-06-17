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
import os


class DistribuGen:
    """

    This class is used to create distributions of the known NEA population and generate samples from them. The class
    first queries the NASA's Jet Propulsion Laboratory Small-Body Database Lookup to get all known NEA's, and publishes
    it to a .csv file with Pandas. This file is then used to extract information from JPL Horizons interface which
    would be pertinent to the generation of a synthetic asteroid image, and saves the file as a Pandas database .csv
    file. The horizons file is then used to fit distributions of the columns in the file using the Fitter package and
    SciPy. The number of random samples can be specified to be drawn from these fitted distributions.
    Each random sample can be used to generate a single synthetic asteroid image.

    Attributes:
        file_path_pre: This is the filepath of the file containing the Small-Body Database Lookup database.
                       If specified, the generator will not query the online database, instead will use the file
                       specified.
        ssb_names: This is a list that specified the column names of the Small-Body Database Lookup. If not specified,
                   the names will be assigned to the minimum required for synthetic asteroid generation.
        h_names: This is a list of column names that should be present in the file after querying JPL horizons. If not
                 specified, the names of the default ssb file plus some required parameters for synthetic asteroid
                 generation will be used.
        dist_names: This is a list of names of the resultant distributions that you would like fitted. If not specified,
                    the minimum set for proper synthetic asteroid generation will be used.
        file_path_post: If specified, the generator bypasses querying JPL Horizons, and instead uses the file at this
                        location. If not specified, this is the file path of the resultant file from querying JPL
                        Horizons.
        ssb_data: The Pandas Dataframe where the small body database lookup is stored.
        h_data: The Pandas Dataframe where the horizons database is stored.
        dist_list: A list of lists that holds the results of the distribution fitting procedure. It is formed by:
                   [ [Parameter name (string), Best distribution name (string), Parameters (tuple)], ...
                   ['Paramter n', '', ()]]
        options: The default option is 's' for standard, checks all distributions for fitting, does visualization,
                and generates samples. The 'q' parameter is for quick, only fitting against common distributions, the
                'v' parameter is for visualization, where if 'v' is specified, visualizations are NOT included. If 'm'
                is specified, then samples are NOT generated. The 'q', 'm' and 'v' parameters can be specified in
                combinations but the exact results for each combination is not verified.
        samples: Holds the final Pandas dataframe of samples randomly drawn from distributions, to be outputted in a
                .csv file.
        dist_range: for some distributions, like the apparent motion, there may be some range over which you want the
                   distribution to be fitted, as there may be some outlier data from JPL Horizons. It is specified
                   as a list of lists: [[min parameter 1, max parameter 1], ..., [min parameter n, max parameter n]].
                   If not specified it will use the entire range (except for apparent motion, where it cuts off at
                   around 100 deg/day)
        obs_loc: The MPC designated observatory location that works with JPL Horizons, default is geocentric Earth (500)
    """

    def __init__(self, fp_pre='NEA_database_ssb.csv', fp_post='NEA_database_hor.csv', fp_final='NEA_distribution.csv',
                 cnames_ssb=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y',
                             'data_arc', 'n_obs_used', 'H', 'first_obs', 'epoch'],
                 cnames_horizon=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used',
                                 'H', 'first_obs', 'epoch', 'omega', 'obs. ast. dist.', 'sun. ast. dist.',
                                 'phase angle'],
                 cnames_dist=['H', 'omega', 'r_as', 'r_ao', 'alpha'], options='s', dist_range=None, observer='500'):
        """

        :param fp_pre: This is the filepath of the file containing the Small-Body Database Lookup database.
                       If specified, the generator will not query the online database, instead will use the file
                       specified.
        :param fp_post: If specified, the generator bypasses querying JPL Horizons, and instead uses the file at this
                        location. If not specified, this is the file path of the resultant file from querying JPL
                        Horizons.
        :param cnames_ssb: This is a list that specified the column names of the Small-Body Database Lookup. If not
                           specified, the names will be assigned to the minimum required for synthetic asteroid
                           generation.
        :param cnames_horizon: This is a list of column names that should be present in the file after querying JPL horizons. If not
                               specified, the names of the default ssb file plus some required parameters for synthetic asteroid
                               generation will be used.
        :param cnames_dist: This is a list of names of the resultant distributions that you would like fitted. If not
                            specified, the minimum set for proper synthetic asteroid generation will be used. Do not
                            include streak orientation and width in these names, as they are later tacked on from
                            sampling uniform distributions
        :param options: The default option is 's' for standard, checks all distributions for fitting, does
                        visualization, and generates samples. The 'q' parameter is for quick, only fitting against
                        common distributions, the 'v' parameter is for visualization, where if 'v' is specified,
                        visualizations are NOT included. If 'm' is specified, then samples are NOT generated.
                        The 'q', 'm' and 'v' parameters can be specified in combinations but the exact results for each
                        combination is not verified.
        :param dist_range: for some distributions, like the apparent motion, there may be some range over which you want
                           the distribution to be fitted, as there may be some outlier data from JPL Horizons. It is
                           specified as a list of lists: [[min parameter 1, max parameter 1], ...,
                           [min parameter n, max parameter n]]. If not specified it will use the entire range
                           (except for apparent motion, where it cuts off at around 100 deg/day)
        :param observer: the observer location as specified by MPC that works with JPL Horizons.
        """

        self.file_path_pre = fp_pre
        self.ssb_names = cnames_ssb
        self.h_names = cnames_horizon
        self.dist_names = cnames_dist
        self.file_path_post = fp_post
        self.ssb_data = None
        self.h_data = None
        self.dist_list = None
        self.options = options
        self.samples = None
        self.dist_range = dist_range
        self.obs_loc = observer
        self.final_path = fp_final
        return

    def database_parser_ssb(self):
        """
        Reads the file containing small body database (separated by commas) specified in the 'file_path_pre' attribute,
        which is initialized by the 'fp_pre' parameter. The column names are specified in the 'ssb_names' attribute,
        which is initialized by the 'cnames_ssb' parameter. Assigned to the 'ssb_data' attribute.
        :return:
        """
        self.ssb_data = pd.read_csv(self.file_path_pre, sep=',', header=0,
                                    names=self.ssb_names)
        return

    def database_parser_h(self):
        """
        Reads the file containing the results of horizon query (separated by commas) specified in the 'file_path_post'
        attribute, which is initialized by the 'fp_post' parameter. The column names are specified in the 'h_names'
        attribute, which is initialized by the 'cnames_horizon' parameter. Assigned to the 'h_data' attribute.
        :return:
        """
        self.h_data = pd.read_csv(self.file_path_post, sep=',', header=0,
                                  names=self.h_names)
        return

    def jpl_querier(self, row):
        """
        Queries JPL horizons online system (requires stable internet connection) and extracts the phase angle,
        apparent motion, observer asteroid distance, sun asteroid distance.

        :param row: this is a row of a particular NEA from the small body database file.

        :return omega: apparent motion (arcsec/hour)
        :return r_oa: the distance between the observer and the asteroid (AU)
        :return r_sa: the distance between the sunand the asteroid (AU)
        :return alpha: the phase angle (deg)
        """
        # Integration step (in days)
        int_step_jpl = 1
        int_step_jpl_unit = 'd'

        # Observation locations
        earth_obs = self.obs_loc  # Geocentric earth by default
        sun_obs = '500@10'  # for heliocentric components

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
        # Heliocentric data
        ##############################################

        nea_helio = Horizons(id=nea_name, location=sun_obs,
                             epochs={'start': start_time, 'stop': end_time,
                                     'step': str(int_step_jpl) + int_step_jpl_unit})

        nea_helio_state = nea_helio.vectors()
        nea_helio_x = nea_helio_state[0]['x']
        nea_helio_y = nea_helio_state[0]['y']
        nea_helio_z = nea_helio_state[0]['z']

        # phase angle
        alpha = nea_eph[0]['alpha']  # in degrees
        # apparent rate of motion
        omega = np.sqrt(ra_rate ** 2 + dec_rate ** 2)  # arcsec/hour
        # observer asteroid distance
        r_oa = np.linalg.norm([nea_geo_x, nea_geo_y, nea_geo_z])  # au
        # sun asteroid distance
        r_sa = np.linalg.norm([nea_helio_x, nea_helio_y, nea_helio_z])  # au

        return omega, r_oa, r_sa, alpha

    def fit_distribution(self):
        """
        Fits a distribution using the Fitter package, requires a proper horizons file specified. Will determine the
        five best performing distributions according to the sumsquare error, and plot each distribution with the best
        fits if specified, as well as print out best distribution names, results, and the computed parameters of the
        distributions. The last two parameters are usually location and scale parameters of the distribution. If Nan
        values are present in the database, those are ignored in the distribution fitting.
        :return:
        """
        summaries = []
        paramss = []
        dist_ranges = []

        # get dataset
        self.database_parser_h()
        data = self.h_data
        col_names = self.dist_names

        # Filter out non-finite data
        for idx, col_name in enumerate(col_names[0:5]):

            finite_values = ~data[col_name].isna() & data[col_name].apply(np.isfinite)
            # Filter the DataFrame to show rows with non-finite values
            rows_with_non_finite = data[finite_values]
            d = rows_with_non_finite[col_name].values
            if col_name == 'omega':
                maxi = 100 * 3600 / 24
            else:
                maxi = np.max(d)
            if 'q' in self.options:
                f = Fitter(d, distributions=get_common_distributions(), xmax=maxi)
            else:
                f = Fitter(d, xmax=maxi)
            f.fit()
            if 'v' in self.options:
                summary = f.summary(plot=False)
            else:
                fig = plt.figure()
                summary = f.summary()
                column_names = ['Absolute Magnitude $H$', 'Apparent Motion $\omega$ (arcsec/h)',
                                'Observer Asteroid Distance $r_{oa}$ (AU)', 'Sun Asteroid Distance $r_{sa}$ (AU)',
                                'Phase Angle $\\alpha$ (deg)']
                plt.xlabel(column_names[idx])
                plt.ylabel('Frequency')
                plt.show()
            summaries.append(summary)
            params = f.fitted_param[summary.index[0]]
            paramss.append(params)
            dist_ranges.append([np.min(d), np.max(d)])

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
        self.dist_range = dist_ranges
        return

    def sample_gen(self, total_samples, ratio):
        """
        Generates a specified number of random samples from the specified distribution names in 'dist_list'. The
        ranges will be limited by the 'dist_range' parameter. A .csv file is outputted with corresponding samples.
        :param num_samples: The number of samples to be generated.
        :return:
        """
        pre_append = []
        num_samples = int(total_samples * ratio)
        for idx, dist in enumerate(self.dist_list[0:5]):
            # unpack
            # col_name = dist[0]
            best_dist = dist[1]
            params = dist[2]

            # Check if the distribution name is valid
            if hasattr(sp, best_dist):
                # Get the distribution object using getattr()
                dist = getattr(sp, best_dist)
                range_min = self.dist_range[idx][0]
                range_max = self.dist_range[idx][1]

                samples = []
                while len(samples) < num_samples:
                    # Generate extra samples until desired number is reached
                    extra_samples = dist.rvs(*params, size=int(num_samples / 10))
                    samples.extend(extra_samples[(extra_samples >= range_min) & (extra_samples <= range_max)])

                # Randomly select num_samples from the generated samples
                random_variable = np.random.choice(samples, num_samples, replace=False)
                pre_append.append(random_variable)

            else:
                print("Invalid distribution name")
                return None

        self.samples = pd.DataFrame(np.array(pre_append).T,
                                    columns=self.dist_names[0:5])

        # for streak orientation and width, draw from uniform distribution and known gaussian distribution
        self.samples['Theta'] = np.random.uniform(0, 360, size=num_samples)
        self.samples['Sigma_g'] = np.random.uniform(0.1, 3, size=num_samples)
        self.samples['g_12'] = np.random.choice([0.58, 0.47], size=num_samples)  # asteroid types c and s
        self.samples['Asteroid Present'] = [True for idx in range(0, num_samples)]

        full_column_names = self.dist_names.copy()
        false_data = pd.DataFrame(np.nan, index=np.arange(total_samples - num_samples), columns=full_column_names)
        false_data['Asteroid Present'] = False
        self.samples = pd.concat([self.samples, false_data], ignore_index=True)
        self.samples.to_csv(self.final_path, sep=',', header=True, index=False)
        return

    @staticmethod
    def parse_name(full_name):
        """
        Takes in a full name from the small body database file and extracts the asteroid name in parentheses to use for
        querying JPL Horizons
        :param full_name: The full_name column from the small body database
        :return: the asteroid name in parentheses.
        """
        matches = re.findall(r"\((.*?)\)", full_name)
        return matches[0]

    def generate(self, num_samples=1, ratio=1.):
        """
        For generate a random number of samples according to the num_samples specified. First, if a file is not
        specified, the small body database is queried for known NEAs, and a .csv file is produced of a comma separated
        Pandas dataframe. This file is then used to query JPL Horizons in order to extract relevant information for
        synthetic asteroid generation with respect to the specified observer, and another .csv file containing a comma
        separated Pandas dataframe is outputted. If this file is specified, querying is skipped, and the file is read
        directly. Then the functin fits distributions to each column in the horizons dataset, and extracts relevant
        distribtuion parameters. With the parameters the function generates num_samples random samples, and outputs a
        comma separated .csv file Pandas dataframe
        :param num_samples: the number of samples to generate
        :return:
        """

        # figure parameters
        sns.set(rc={'figure.figsize': (8, 6)})
        sns.set_style('ticks')
        sns.set_context("paper", font_scale=2)

        if os.path.exists(self.file_path_pre):
            pass
        else:
            # query SSB
            # generate file, set file path name, column names
            pass

        if os.path.exists(self.file_path_post):
            pass
        else:
            # query jpl horizons
            self.horizons_query()
        self.fit_distribution()
        if 'm' in self.options:
            pass
        else:
            self.sample_gen(num_samples, ratio)

        if 'v' in self.options:
            pass
        else:
            plt.show()

        return

    def horizons_query(self):
        """
        Query JPL horizons by adding a new row to a Pandas database stored as a .csv file. Parameters of resultant file
        include: 'full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used', 'H', 'first_obs',
        'epoch', 'omega', 'obs. ast. dist.', 'sun. ast. dist.', 'phase angle'
        :return:
        """
        self.database_parser_ssb()

        for idx, row in self.ssb_data.iterrows():

            print("Querying: " + row['full_name'])
            try:
                new_data = self.jpl_querier(row)
                new_row = pd.DataFrame(
                    {'full_name': row['full_name'], 'a': row['a'], 'e': row['e'],
                     'i': row['i'], 'om': row['om'], 'w': row['w'], 'q': row['q'],
                     'ad': row['ad'], 'per_y': row['per_y'],
                     'data_arc': row['data_arc'], 'n_obs_used': row['n_obs_used'], 'H': row['H'],
                     'first_obs': row['first_obs'], 'epoch': row['epoch'], 'omega': new_data[0],
                     'obs. ast. dist.': new_data[1],
                     'sun. ast. dist.': new_data[2], 'phase angle': new_data[3]}, index=[1])

                new_row.to_csv(self.file_path_post, sep=',', mode='a', header=False, index=False)
            except ValueError:
                print("No ephemeris for target")


if __name__ == '__main__':
    # options - 'v': do not include visualization
    #           'q': quick fit of distribution looking at only common ones
    #           's': standard full package (default)
    #           'm': do not generate samples
    dis_gen = DistribuGen(fp_pre='NEA_database_ssb.csv', fp_post='NEA_database_hor.csv',
                          cnames_dist=['H', 'omega', 'obs. ast. dist.', 'sun. ast. dist.', 'phase angle'],
                          options='qv')
    number_of_samples = 100
    dis_gen.generate(number_of_samples, 0.1)

    # data = pd.read_csv('NEA_distribution_samples_10000000.csv', sep=',', header=0,
    #                    names=['H', 'omega', 'obs. ast. dist.', 'sun. ast. dist.', 'phase angle'])
    # print(data)
    # dis_gen.database_parser_h()
    # dis_gen.database_parser_ssb()
    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns
    # plt.hist(dis_gen.h_data['omega'], bins=20000)
    # print(dis_gen.h_data['omega'])
    # plt.show()
