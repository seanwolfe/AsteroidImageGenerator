import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class SignalGen:

    def __init__(self, dt=5, sig_rn=1.6, i_dark=0.5, v_bg=20.5, v_0=22.1, pixel_scale=1.26, sig_g=1.274, omega=2160,
                 r_sa=1, r_oa=0.2, h=24, g_12=0.58, theta=45, n_frames=100, alpha=0, sig_bg=0.223):
        self.dt = dt
        self.sig_rn = sig_rn
        self.i_dark = i_dark
        self.v_bg = v_bg
        self.v_0 = v_0
        self.pixel_scale = pixel_scale
        self.fwhm = 2 * np.sqrt(2 * np.log(2)) * sig_g  # in arcsec
        self.omega = omega / 3600  # arcsec / sec
        self.r_sa = r_sa
        self.r_oa = r_oa
        self.h = h
        if g_12 >= 0.2:
            self.g_1 = 0.9529 * g_12 + 0.02162
            self.g_2 = -0.6125 * g_12 + 0.5572
        else:
            self.g_1 = 0.7527 * g_12 + 0.06164
            self.g_2 = -0.9612 * g_12 + 0.6270
        self.theta = theta
        self.n_frames = n_frames
        self.alpha = alpha
        self.phi_1, self.phi_2, self.phi_3 = self.calc_phis()
        self.v_s = self.apparent_magnitude_calc()
        self.snr = self.snr_calc()
        self.sig_bg = sig_bg

        return

    def calc_phis(self):
        alphas_phi_12 = [0, 7.5, 30, 60, 90, 120, 150, 180]
        phi_1_values = [1, 7.5e-1, 3.3486016e-1, 1.3410560e-1, 5.1104756e-2, 2.1465687e-2, 3.6396989e-3, 0]
        phi_2_values = [1, 9.25e-1, 6.2884169e-1, 3.1755495e-1, 1.2716367e-1, 2.2373903e-2, 1.6505689e-4, 0]
        alphas_phi_3 = [0, 0.3, 1, 2, 4, 8, 12, 20, 30, 60, 90, 180]
        phi_3_values = [1, 8.3381185e-1, 5.7735424e-1, 4.2144772e-1, 2.3174230e-1, 1.0348178e-1, 6.1733473e-2,
                        1.6107006e-2, 0, 0, 0, 0]
        phi_1 = CubicSpline(alphas_phi_12, phi_1_values,
                            bc_type=((1, -6 / np.pi * 2 * np.pi / 360), (1, -9.1328612e-2 * 2 * np.pi / 360)))
        phi_2 = CubicSpline(alphas_phi_12, phi_2_values,
                            bc_type=((1, -9 / (5 * np.pi) * 2 * np.pi / 360), (1, -8.6573138e-8 * 2 * np.pi / 360)))
        phi_3 = CubicSpline(alphas_phi_3, phi_3_values,
                            bc_type=((1, -1.0630097e-1 * 2 * np.pi / 360), (1, 0 * 2 * np.pi / 360)))
        return phi_1(self.alpha), phi_2(self.alpha), phi_3(self.alpha)

    def apparent_magnitude_calc(self):

        psi = self.g_1 * self.phi_1 + self.g_2 * self.phi_2 + (1 - self.g_1 - self.g_2) * self.phi_3
        v_s = self.h + 5 * np.log10(self.r_sa * self.r_oa) - 2.5 * np.log10(psi)
        return v_s

    def snr_calc(self):

        # calc terms in overall SNR
        t1 = np.sqrt(self.dt) / (np.sqrt(np.pi / (2 * np.log(2))) * self.fwhm)
        t2 = np.power(10, -0.4 * (self.v_s - self.v_0)) / np.sqrt(
            self.pixel_scale ** 2 * np.power(10, -0.4 * (self.v_bg - self.v_0)) + self.i_dark)
        t3 = np.power(1 + self.sig_rn ** 2 / ((
                                                      self.pixel_scale ** 2 * np.power(10, -0.4 * (
                                                          self.v_bg - self.v_0)) + self.i_dark) * self.dt), -0.5)

        # for reduction function
        s = self.omega * self.dt / (2 * self.fwhm / np.sqrt(2 * np.log(2)))
        if s >= 2:
            t4 = np.sqrt(np.pi) / (2 * s)
        else:
            t4 = 1 / (1 + s ** 2 / 3)

        snr = t1 * t2 * t3 * t4

        return snr


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
