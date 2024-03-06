import re

import pandas as pd
from astroquery.jplhorizons import Horizons
import re
from astropy.time import Time
import astropy.units as u
import numpy as np


class DistribuGen:

    # properties

    def __int__(self):
        return

    @staticmethod
    def database_parser(file_path='NEA_database.csv'):
        if file_path == 'NEA_database.csv':
            data = pd.read_csv(file_path, sep=',', header=0,
                               names=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y',
                                      'data_arc', 'n_obs_used', 'H', 'first_obs', 'epoch'])
        elif file_path == 'NEA_database1.csv':
            data = pd.read_csv(file_path, sep=',', header=0,
                               names=['full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y',
                                      'data_arc', 'n_obs_used', 'H', 'first_obs', 'epoch', 'omega', 'obs. ast. dist.',
                                      'sun. ast. dist.', 'phase angle'])

        else:
            data = []
            print("ERROR: Please specify correct NEA database path")
        return data

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

    @staticmethod
    def parse_name(full_name):
        matches = re.findall(r"\((.*?)\)", full_name)
        return matches[0]

    def main_parse(self):
        fp = 'NEA_database.csv'
        fp1 = 'NEA_database1.csv'
        d = self.database_parser(fp)

        for idx, row in d.iterrows():
            if idx > 20389:
                print("Querrying: " + row['full_name'])
                try:
                    new_data = self.jpl_querier(row)
                    new_row = pd.DataFrame(
                        {'full_name': row['full_name'], 'a': row['a'], 'e': row['a'],
                         'i': row['i'], 'om': row['om'], 'w': row['w'], 'q': row['q'],
                         'ad': row['ad'], 'per_y': row['per_y'],
                         'data_arc': row['data_arc'], 'n_obs_used': row['n_obs_used'], 'H': row['H'],
                         'first_obs': row['first_obs'], 'epoch': row['epoch'], 'omega': new_data[0],
                         'obs. ast. dist.': new_data[1],
                         'sun. ast. dist.': new_data[2], 'phase angle': new_data[3]}, index=[1])

                    new_row.to_csv(fp1, sep=',', mode='a', header=False, index=False)
                except ValueError:
                    print("No ephemeris for target")


if __name__ == '__main__':
    disgen = DistribuGen()
    disgen.main_parse()
