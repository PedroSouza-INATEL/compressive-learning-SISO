"""
    Created on Tue Jan 26/2021
    Last rev. on Tue Jan 27/2021
    © 2020 Pedro H. C. de Souza
           Luciano Leonel Mendes
"""
import math as mt
import sys
import time
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from signal_processing import MonteCarlo
from scipy.integrate import dblquad
from scipy.special import erfc
start_time = time.time()

# Initialize system model parameters
MC_RUNS = 10**5
N_POINTS = 21
N = 1024
M = np.array([N, 0.25 * N], dtype=int)
comp_rate = N // M
n_curves = np.size(M)
C = 3
snr = np.linspace(0, 40, N_POINTS)
gamma = 10**(snr / 10)
snr_theory = np.linspace(snr.min(), snr.max(), 100)
gamma_theory = 10**(snr_theory / 10)

# Define channel path delays
tau = np.array([0, 8, 16, 32]) * 2

# Initialize estimation scheme parameters
power = (1 / np.mean(tau)) * np.exp(-tau / np.mean(tau))
power /= np.sum(power)
phi = -2 * mt.pi * tau / (N - 1)
tau_mean = np.sum((tau / (N - 1)) * power)
tau_std = np.sqrt(np.sum(((tau / (N - 1)) - tau_mean)**2 * power))
bwc = 1 / (0.5 * tau_std)


# Modules used in the analytical performance computation
def erfc_y(y, C):
    return 1 - (1 - 0.5 * erfc(y / mt.sqrt(2)))**(C - 1)


def exp_g(y, g):
    return mt.exp(-0.5 * (y - mt.sqrt(2 * g))**2)


def chi2(g, gamma_theory):
    return (1 / gamma_theory) * mt.exp(-g / gamma_theory)


# Neural network hyperparameters initialization
N_TRAIN = 10**4
N_LAYER = 10
N_NEURON = np.tile(200, N_LAYER)
FUNC_NEURON = 'relu'
SOLVER = 'adam'
INIT_ETA = 10**-3

# Generate multiple classes of signal vectors, comprising of random iid
# strings of levels
data_signal = np.zeros([C, N])
for i in range(C):
    # Generate a signal vector with bernoulli ±1 levels
    data_signal[i] = 2 * (np.random.uniform(0, 1, N) >= 0.5) - 1
    # plot text box label
    DATA_LABEL = r'$P(s(n)^{(i)}_j = \pm 1) = 1/2$'

    # Generate a signal vector with Gaussian distributed levels
    # data_signal[i, 0] = np.random.randn(N)
    # DATA_LABEL = r'$s(n_fft)^{(i)}_j \sim N(0,1)$'

# Compute the minimum distance separation
dist = np.linalg.norm(data_signal[0] - data_signal[1])
for i in range(C - 1):
    for j in range(C - i - 1):
        temp_d = np.linalg.norm(data_signal[i] - data_signal[j + i + 1])

        if temp_d < dist:
            dist = temp_d

# Initialize variables
pe_ml = np.zeros([N_POINTS, n_curves])
pe_nn = np.zeros([N_POINTS, n_curves])
ml = np.zeros([C, MC_RUNS])

# Execute Monte Carlo iterations for each curve
for i in range(n_curves):
    # Compute orthonormal sensing matrix. If a vector signal is
    # transmitted uncompressed, then 'A' = I (identity matrix)
    if M[i] < N:
        A = np.random.randn(M[i], N) / mt.sqrt(N)
    else:
        A = np.eye(N)

    # Store beforehand all possible compressed data signals that can be transmitted
    comp_signal = data_signal @ A.T

    # Calculate number of pilots based upon the coherence band
    n_sc_all = 2**(np.arange(1, np.log2(M[i]) + 1))
    n_sc = min(n_sc_all[np.flatnonzero(np.floor(M[i] / bwc) < n_sc_all)]).astype(int)

    # Compute AWG noise standard variation (systems' and MMSEs')
    snr_uniform = np.random.uniform(snr.min(), snr.max(), N_TRAIN)
    std = np.sqrt(dist**2 / 10**(snr_uniform / 10))
    gamma_mmse = 10**(snr_uniform / 10)

    # Invoke class that simulates mutiple data signals transmissions, channel
    # impairments and MMSE estimation
    params = MonteCarlo(N_TRAIN, M[i], C, n_sc)
    rx_signal, class_idx, coef_interp = \
        params.signal_process(comp_signal, std, gamma_mmse,
                              tau // comp_rate[i], power, phi)

    # Received signal ready for neural network input
    rx_signal = np.c_[np.real(rx_signal), np.imag(rx_signal)]

    # Estimated channel coefficients also ready for neural network input
    ref_signal = coef_interp.T
    ref_signal = np.c_[np.real(ref_signal), np.imag(ref_signal)]

    # Build and train neural network model
    clf = MLPClassifier(N_NEURON, FUNC_NEURON, learning_rate_init=INIT_ETA,
                        random_state=42)  # random_state=42
    clf.fit(np.c_[rx_signal, ref_signal], class_idx)

    # Compute Monte Carlo iterations for each curve point
    for j in range(N_POINTS):
        # Generate the progress bar for evaluation purposes. It does not have
        # any impact on the results
        sys.stdout.write('\r')
        if n_curves > 1:
            sys.stdout.write("[{:{}}] {:.1f}%  ".format("=" * i, n_curves - 1,
                                                        (100 / (n_curves - 1) * i)))
        sys.stdout.write("[{:{}}] {:.1f}%  ".format("=" * j, N_POINTS - 1,
                                                    (100 / (N_POINTS - 1) * j)))
        sys.stdout.flush()

        # Compute AWG noise standard variation (systems' and MMSEs')
        std = np.tile(mt.sqrt(dist**2 / gamma[j]), MC_RUNS)
        gamma_mmse = np.tile(gamma[j], MC_RUNS)

        # Clear cumulative error counter
        ERR_ML = 0
        ERR_NN = 0

        # Invoke class that simulates mutiple data signals transmissions, channel
        # impairments and MMSE estimation
        params = MonteCarlo(MC_RUNS, M[i], C, n_sc)
        rx_signal, class_idx, coef_interp = \
            params.signal_process(comp_signal, std, gamma_mmse,
                                  tau // comp_rate[i], power, phi)

        prob = np.zeros([MC_RUNS, C])
        ref_signal = coef_interp.T

        # Compute the maximum likelihood statistic on the received signal
        for p in range(C):
            ml[p] = np.linalg.norm(rx_signal - comp_signal[p] * ref_signal, axis=1)**2

        ref_signal = np.c_[np.real(ref_signal), np.imag(ref_signal)]

        # Classes probabilities predicted by the neural network
        prob = clf.predict_proba(np.c_[np.real(rx_signal),
                                       np.imag(rx_signal), ref_signal])

        # Error count
        ERR_ML = np.sum(np.argmin(ml, 0) != class_idx)
        ERR_NN = np.sum(np.argmax(prob, 1) != class_idx)

        # Estimate the probability of error
        pe_ml[j, i] = ERR_ML / MC_RUNS
        pe_nn[j, i] = ERR_NN / MC_RUNS

# Estimate runtime for evaluation purposes
print("\n %f seconds" % (time.time() - start_time))

# Plot results
fig, ax = plt.subplots()

pe_theory = [None] * n_curves
for i in range(n_curves):
    # For binary classes (i.e. 'C' = 2), analytical curves of probability
    # of error are computed according to (6.208) in [3, p. 575], otherwise
    # the numerical approximation (6.207) is used.
    # Originally (6.207) and (6.208) were developed considering symbols'
    # energies. However, we are considering euclidian distances among symbols
    # and not their energy as is the standard, hence the 2 dividing gamma_theory.
    # Note that, considering standard symbol energy $\sqrt{Eb}$,
    # $d_{min} = $\sqrt{2Eb}$ for this expression! Since it refers to the M-FSK
    # modulation, for which symbols are mutually orthogonal.
    if C == 2:
        pe_theory[i] = 0.5 * (1 - np.sqrt(gamma_theory / (4 * comp_rate[i] + gamma_theory)))
    else:
        pe_theory[i] = [dblquad(lambda g, y:(1 / mt.sqrt(2 * mt.pi)) * exp_g(y, g) *
                                chi2(g, gamma_theory[j] / (2 * comp_rate[i])) * erfc_y(y, C),
                                -mt.inf, mt.inf, lambda g: 0, lambda g: mt.inf)[0]
                        for j in range(gamma_theory.size)]

    plt.plot(snr_theory, pe_theory[i], '-b', linewidth=1.25, markersize=3.75,
             label='Theory', zorder=10)

    plt.plot(snr, pe_ml[:, i], ('-%ck' % Line2D.filled_markers[i]),
             linewidth=1.25, fillstyle='none', markersize=3.75,
             label=(r'$\frac{M}{N} = %.2f$ (MLD)' % (1 / comp_rate[i])))
    plt.plot(snr, pe_nn[:, i], ('--%cr' % Line2D.filled_markers[i]),
             linewidth=1.25, fillstyle='none', markersize=3.75,
             label=(r'$\frac{M}{N} = %.2f$ (NND)' % (1 / comp_rate[i])))

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={'size': 6})
# plt.legend()

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
TEXTSTR = '\n'.join(('Rayleigh (selective)', DATA_LABEL, r'$N = %.i$ samples' % N,
                     r'$C = %.i$ classes' % C,
                     r'$N_p = %i$ pilots (for M/N = 1)' % (n_sc * comp_rate[-1] + 1),
                     r'$\tau = $' + np.array2string(tau, separator=',')))
TEXTSTR_ = '\n'.join(('MLP parameters:',
                      r'$N_{\mathcal{S}_{TR}} = %.E$' % N_TRAIN,
                      r'$L = %.i$' % N_LAYER,
                      r'$N_\ell = %.i$' % N_NEURON[0],
                      FUNC_NEURON,
                      r'$\eta_{init} = %.E$' % INIT_ETA,
                      SOLVER))
plt.text(0.025, 0.025, TEXTSTR, transform=ax.transAxes, fontsize='x-small',
         verticalalignment='bottom', bbox=props)
plt.text(0.025, 0.3, TEXTSTR_, transform=ax.transAxes, fontsize='x-small',
         verticalalignment='bottom', bbox=props)

plt.axis([snr.min(), snr.max(), 10**-4, 1])
plt.semilogy()
plt.xlabel(r'$10\log_{10} (\Gamma)$')
plt.ylabel(r'$P_e$')
plt.title('MLD and NND detection performance')
plt.grid(which='major', linestyle='--')
plt.grid(which='minor', linestyle='--', linewidth=0.5)
