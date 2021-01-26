import math as mt
import numpy as np
from scipy import interpolate


class MonteCarlo():
    def __init__(self, MC_RUNS, M, C, n_sc):
        self.mc_runs = MC_RUNS
        self._m = M
        self._c = C
        self.n_sc = n_sc

        # Initialize variables
        self._coef_ch = np.zeros(M, dtype=complex)
        # self._coef_fft = np.zeros(n_fft, dtype=complex)
        self._coef_interp = np.zeros([M, MC_RUNS], dtype=complex)
        self.rx_signal = np.zeros([MC_RUNS, M], dtype=complex)

    def signal_process(self, comp_signal, std, gamma_mmse, tau, power, phi):
        # Define indexes of subcarriers transmitting pilot tones
        # subcarrier_k = np.arange(0, self._m + 1, self._m // self.n_sc)
        # if self._m == self.n_sc:
        #     subcarrier_k = subcarrier_k[:-1]
        # else:
        #     subcarrier_k[-1] = self._m - 1
        #     self.n_sc += 1

        n_fft = 2 * self._m

        # Choose which class of vector signal to transmit. Classes occurrences
        # are random (i.i.d) and equiprobable
        class_idx = np.random.randint(self._c, size=self.mc_runs)

        # Initialize complex Gaussian channel coefficients (Rayleigh fading)
        coef = (1 / mt.sqrt(2)) * (np.random.randn(self.mc_runs) +
                                   1j * np.random.randn(self.mc_runs))

        # Execute Monte Carlo loop
        for k in range(self.mc_runs):
            # Initialize complex AWG noise at the receiver
            noise = (std[k] / mt.sqrt(2)) * (np.random.randn(self._m) +
                                             1j * np.random.randn(self._m))

            # Initialize AWG noise that corrupts MMSE estimation. Its standard
            # deviation does not depend on 'dist', since pilot tones were chosen
            # to have unit power without loss of generality
            # noise_mmse = (mt.sqrt(1 / gamma_mmse[k]) / mt.sqrt(2)) * \
            #     (np.random.randn(self.n_sc) + 1j *
            #      np.random.randn(self.n_sc))

            # Compute frequency selective channel response via FFT
            # of the channel impulse response
            self._coef_ch[tau] = np.sqrt(power) * np.exp(1j * phi) * coef[k]
            coef_fft = np.fft.fft(self._coef_ch, n_fft)

            # Perform MMSE estimation of channel coefficients based on
            # the pilots transmitted by subcarriers
            # coef_mmse = (1 / (1 + 1 / gamma_mmse[k])) * (coef_fft[subcarrier_k] + noise_mmse)
            # coef_mmse = coef_fft[subcarrier_k, m] + noise_mmse[:, :, m]

            # Compute interpolation of the estimated channel coefficients
            # func = interpolate.interp1d(subcarrier_k, coef_mmse, kind=1)
            # self._coef_interp[:, k] = func(np.arange(0, n_fft // 2))
            self._coef_interp[:, k] = coef_fft[:n_fft // 2]

            # Corrupt the received signal with channel impairments
            self.rx_signal[k] = comp_signal[class_idx[k]] * coef_fft[:n_fft // 2] + noise

        return self.rx_signal, class_idx, self._coef_interp
