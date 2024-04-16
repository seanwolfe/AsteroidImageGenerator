import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class SignalGen:
    """

    This class computes the expected signal level an asteroid with specific properties would produce when viewed from
    a telescope with specific properties. The derivation follows that of Zhai et al. 2024 for the SNR, but does not
    include the sqrt(N_f) factor that improves SNR when using sythetic tracking. Instead the SNR is estimated for
    an asteroid based on the streaked motion it would produce in a single exposure.

    Attributes:
        dt: the single frame exposure time (seconds)
        sig_rn: the read noise specification of the telescope in electrons per pixel
        i_dark: the dark current specification of the telescope in electrons per second pixel
        v_bg: the background sky brightness for the telescope in magnitude per arcsec^2
        v_0: the telescope's system zero point in magnitudes
        pixel_scale: the pixel scale of the telescope in arcsec per pixel

    Other relevant parameters
        fwhms: the full-width half maximum of the telescope?/asteroid?, related to the standard deviation of a Gaussian
        representing a streak to be convolved with the asteroid, in pixels (internally converted to arcsec)
        omegas: the apparent motion of the asteroid in arcsec/hour (internally converted to arcsec/sec)
        r_sa: the asteroid sun distance in AU
        r_oa: the asteroid observer distance in AU
        h: the absolute magnitude of the asteroid
        g_1 and g_1: the slope parameters for the three parameter system specified in Muinonen et al. 2010, and
         computed from the constant representing the asteroid type g_12 (see Veres et al. 2015)
        alpha: the phase angle in degrees
        phi_1, phi_2 and phi_3: the phase functions from Muinonen et al. 2010
        v_s: the apparent magnitude of the asteroid
        snr: the expected SNR of the asteroid
        sig_bg: the standard deviation of the background noise in the image to be implanted
        sig_level: the expected signal level of the asteroid in order to create such a streak

    """

    def __init__(self, configs):
        """

        dt: the single frame exposure time (seconds)
        sig_rn: the read noise specification of the telescope in electrons per pixel
        i_dark: the dark current specification of the telescope in electrons per second pixel
        v_bg: the background sky brightness for the telescope in magnitude per arcsec^2
        v_0: the telescope's system zero point in magnitudes
        pixel_scale: the pixel scale of the telescope in arcsec per pixel
        sig_g: the standard deviation of a Gaussian representing a streak to be convolved with the asteroid,
        in pixels (internally converted to arcsec, and to full-width half maximum)
        omega: the apparent motion of the asteroid in arcsec/hour (internally converted to arcsec/sec)
        r_sa: the asteroid sun distance in AU
        r_oa: the asteroid observer distance in AU
        h: the absolute magnitude of the asteroid
        g_12: constant representing the asteroid type (see Veres et al. 2015), to compute g_1 and g_2
        alpha: the phase angle in degrees
        sig_bg: the standard deviation of the background noise in the image to be implanted

        Other Attributes:
        fwhm: the full-width half maximum of the telescope?/asteroid?, related to the standard deviation of a Gaussian
        g_1 and g_1: the slope parameters for the three parameter system specified in Muinonen et al. 2010, and
         computed from the constant representing the asteroid type g_12 (see Veres et al. 2015)
        phi_1, phi_2 and phi_3: the phase functions from Muinonen et al. 2010
        v_s: the apparent magnitude of the asteroid
        snr: the expected SNR of the asteroid
        sig_bg: the standard deviation of the background noise in the image to be implanted
        sig_level: the expected signal level of the asteroid in order to create such a streak
        """

        #
        self.dt = configs['dt']
        self.sig_rn = configs['sig_rn']
        self.i_dark = configs['i_dark']
        self.v_bg = configs['v_bg']
        self.v_0 = configs['v_0']
        self.pixel_scale = configs['pixel_scale']  # in arcsec/pixel
        self.configuration = configs

        if 't' in self.configuration['options']:
            pass
        else:
            self.dist_data = pd.read_csv(configs['distribution_file_name'], sep=',', header=0,
                                         names=configs['distribution_file_columns'])
            self.stack_data = pd.read_csv(configs['stack_file_name'], sep=',', header=0,
                                          names=configs['stack_file_columns'])

            # see Muinonen et al 2010 for details on g_1 and g_2 from g_12
            self.g_1 = pd.Series(self.dist_data['g_12']).apply(
                lambda g_12: 0.9529 * g_12 + 0.02162 if g_12 >= 0.2 else 0.7527 * g_12 + 0.06164)
            self.g_2 = pd.Series(self.dist_data['g_12']).apply(
                lambda g_12: -0.6125 * g_12 + 0.5572 if g_12 >= 0.2 else -0.9612 * g_12 + 0.6270)

        return

    def calc_phis(self, alphas):
        """
        :param alphas: the phase angles to be used for calculation of phase functions (in degrees)
        fit cubic splines on values obtained for the various phase functions phi_1, phi_2 and phi_3 of Muinonen et al.
        2010. Also given the specified boundary conditions on the derivatives.
        :return: Returns the phase values phi_1(alpha), phi_2(alpha) and phi_3(alpha) at the specified phase angles.
        """
        # points of the cubic splines to be fit
        alphas_phi_12 = self.configuration['alphas_phi_12']
        phi_1_values = self.configuration['phi_1_values']
        phi_2_values = self.configuration['phi_2_values']
        alphas_phi_3 = self.configuration['alphas_phi_3']
        phi_3_values = self.configuration['phi_3_values']

        # fit the cubic splines with boundary conditions specified based on the first order requirements
        phi_1 = CubicSpline(alphas_phi_12, phi_1_values,
                            bc_type=((1, self.configuration['phi_1_derivative_initial'] * 2 * np.pi / 360),
                                     (1, self.configuration['phi_1_derivative_end'] * 2 * np.pi / 360)))
        phi_2 = CubicSpline(alphas_phi_12, phi_2_values,
                            bc_type=((1, self.configuration['phi_2_derivative_initial'] * 2 * np.pi / 360),
                                     (1, self.configuration['phi_2_derivative_end'] * 2 * np.pi / 360)))
        phi_3 = CubicSpline(alphas_phi_3, phi_3_values,
                            bc_type=((1, self.configuration['phi_3_derivative_initial'] * 2 * np.pi / 360),
                                     (1, self.configuration['phi_3_derivative_end'] * 2 * np.pi / 360)))
        return phi_1(alphas), phi_2(alphas), phi_3(alphas)

    def apparent_magnitude_calc(self, phi_1_s, phi_2_s, phi_3_s):
        """
        calculate the apparent magnitude of the asteroid following the three parameter phase function, as well as a
        parameter including the distance relationship, the formula is similar to the manner in which JPL Horizons
        computes the apparent magnitude

        :param phi_1_s
        :param phi_2_s
        :param phi_3_s: lists of the phi functions evaluated at specified phase angles (see calc_phis function)

        :return: the apparent magnitude
        """
        psi_s = self.g_1 * phi_1_s + self.g_2 * phi_2_s + (1 - self.g_1 - self.g_2) * phi_3_s
        v_s = self.dist_data['H'] + 5 * np.log10(
            self.dist_data['sun. ast. dist.'] * self.dist_data['obs. ast. dist.']) - 2.5 * np.log10(psi_s)
        return v_s

    def snr_calc(self, v_s_s):
        """
        Computes the SNR given a slightly modified version of equation 5 in Zhai et al. 2024, in order to remove
        dependency on total expsosure time of N_frames and just have it for a single frame.
        :return: signal to noise ratio
        """

        fwhms = 2 * np.sqrt(2 * np.log(2)) * self.pixel_scale * self.dist_data['Sigma_g']
        omegas = self.dist_data['omega'] / 3600

        # calc terms in overall SNR
        t1 = np.sqrt(self.dt) / (np.sqrt(np.pi / (2 * np.log(2))) * fwhms)
        t2 = np.power(10, -0.4 * (v_s_s - self.v_0)) / np.sqrt(
            self.pixel_scale ** 2 * np.power(10, -0.4 * (self.v_bg - self.v_0)) + self.i_dark)
        t3 = np.power(1 + self.sig_rn ** 2 / (
                (self.pixel_scale ** 2 * np.power(10, -0.4 * (self.v_bg - self.v_0)) + self.i_dark) * self.dt),
                      -0.5)

        # for reduction function
        s = pd.DataFrame(omegas * self.dt / (2 * fwhms / np.sqrt(2 * np.log(2))), columns=['reductions'])
        # t4 = s.copy()
        # t4[s['reductions'] >= 2] = np.sqrt(np.pi) / (2 * s)

        t4 = s.applymap(lambda x: np.sqrt(np.pi) / (2 * x) if x >= 2 else 1 / (1 + x ** 2 / 3))

        # terms 3 and 4 represent the sensitivity in Zhai et al. 2024.
        snr = t1 * t2 * t3 * t4['reductions']

        return snr

    def signal_calc(self, snr):
        """
        calculate the expected signal level in an image with an expected snr and a measured background standard
        deviation. Using SNR = S / sqrt(S + B)
        :return: the expected signal level
        """
        snr_vals = snr.values
        big_l = self.dist_data['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
        print(self.stack_data['Stack Mean'])
        print(self.dist_data['Sigma_g'] * 2 * np.sqrt(2 * np.pi) * big_l)
        mean_vals = self.stack_data['Stack Mean'] * self.dist_data['Sigma_g'] * 2 * np.sqrt(2 * np.pi) * big_l
        ones = np.ones_like(snr_vals)
        coefficients = np.array([ones, -snr_vals ** 2, -snr_vals ** 2 * mean_vals]).T
        d = coefficients[:, 1:-1] ** 2 - 4.0 * coefficients[:, ::2].prod(axis=1, keepdims=True)
        roots = -0.5 * (coefficients[:, 1:-1] + [1, -1] * np.emath.sqrt(d)) / coefficients[:, :1]
        return roots[:, 1] + self.stack_data['Stack Mean']

    def signal_calc_test(self, master):
        """
        calculate the expected signal level in an image with an expected snr and a measured background standard
        deviation. Using SNR = S / sqrt(S + B)
        :return: the expected signal level
        """
        if 't' in self.configuration['options']:
            snr_vals = master['Expected SNR'].values
            big_l = master['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
            background_flux = (big_l + 2 * self.configuration['num_sigmas'] * master['Sigma_g']) * (
                        self.configuration['num_sigmas'] * master['Sigma_g']) * master['Stack Mean']
            ones = np.ones_like(snr_vals)
            coefficients = np.array([ones, -snr_vals ** 2, -snr_vals ** 2 * background_flux]).T
            d = coefficients[:, 1:-1] ** 2 - 4.0 * coefficients[:, ::2].prod(axis=1, keepdims=True)
            roots = -0.5 * (coefficients[:, 1:-1] + [1, -1] * np.emath.sqrt(d)) / coefficients[:, :1]
        else:
            snr_vals = master.values
            big_l = self.dist_data['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
            background_flux = (big_l + 2 * self.configuration['num_sigmas'] * self.dist_data['Sigma_g']) * (
                    self.configuration['num_sigmas'] * self.dist_data['Sigma_g']) * self.stack_data['Stack Mean']
            ones = np.ones_like(snr_vals)
            coefficients = np.array([ones, -snr_vals ** 2, -snr_vals ** 2 * background_flux]).T
            d = coefficients[:, 1:-1] ** 2 - 4.0 * coefficients[:, ::2].prod(axis=1, keepdims=True)
            roots = -0.5 * (coefficients[:, 1:-1] + [1, -1] * np.emath.sqrt(d)) / coefficients[:, :1]
        return roots[:, 1]

    def signal_calc_test2(self, master):
        """
        calculate the expected signal level in an image with an expected snr and a measured background standard
        deviation. Using SNR = S / sqrt(S + B)
        :return: the expected signal level
        """
        snr_vals = master['Expected SNR'].values
        background_flux = master['Stack Mean']
        ones = np.ones_like(snr_vals)
        coefficients = np.array([ones, -snr_vals ** 2, -snr_vals ** 2 * background_flux]).T
        d = coefficients[:, 1:-1] ** 2 - 4.0 * coefficients[:, ::2].prod(axis=1, keepdims=True)
        roots = -0.5 * (coefficients[:, 1:-1] + [1, -1] * np.emath.sqrt(d)) / coefficients[:, :1]
        return roots[:, 1]

    def gen_snr_file(self):

        phi_1_s, phi_2_s, phi_3_s = self.calc_phis(self.dist_data['phase angle'])
        v_s_s = self.apparent_magnitude_calc(phi_1_s, phi_2_s, phi_3_s)
        snr = self.snr_calc(v_s_s)
        sig_level = self.signal_calc_test(snr)
        data = {'Expected SNR': snr,
                'Expected Signal': sig_level}
        snr_data = pd.DataFrame(data)
        snr_data.to_csv(self.configuration['snr_file_name'], sep=',', header=True, index=False)
        return


if __name__ == '__main__':
    print("Hello World")
    # p1s = []
    # p2s = []
    # p3s = []
    # iss = []
    # for i in range(0, 180):
    #     print('\n')
    #
    #     signaler = SignalGen(alpha=i)
    #     print(signaler.v_s)
    #     if np.isnan(signaler.phi_1):
    #         p1s.append(p1s[i - 1])
    #     else:
    #         if signaler.phi_1 < 0:
    #             signaler.phi_1 = 0
    #         p1s.append(signaler.phi_1)
    #     if np.isnan(signaler.phi_2):
    #         p2s.append(p2s[i - 1])
    #     else:
    #         if signaler.phi_2 < 0:
    #             signaler.phi_2 = 0
    #         p2s.append(signaler.phi_2)
    #     if np.isnan(signaler.phi_3):
    #         p3s.append(p3s[i - 1])
    #     else:
    #         if signaler.phi_3 < 0:
    #             signaler.phi_3 = 0
    #         p3s.append(signaler.phi_3)
    #
    #     iss.append(i)
    #     print(np.sqrt(signaler.n_frames) * signaler.snr)
    # plt.figure()
    # plt.plot(iss, p1s, label='$\Phi_1(\\alpha)$', color='#1588e6')
    # plt.plot(iss, p2s, label='$\Phi_2(\\alpha)$', color='#8dbbe0')
    # plt.plot(iss, p3s, label='$\Phi_3(\\alpha)$', color='#56748c')
    # plt.xlabel('Phase Angle $\\alpha$ (deg)')
    # plt.ylabel('Phase Function')
    # plt.legend()
    # plt.show()
