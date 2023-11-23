import matlab.engine
import numpy as np
import mne

def MVMD_matlab(signal, alpha, tau, K, DC, init, tol):
    """
    Multivariate Variational Mode Decomposition (MVMD) algorithm for decomposing multivariate or multichannel signals
    into a specified number of modes.

    Parameters:
    -----------
    signal : array-like
        Input multivariate signal to be decomposed. The shape should be (C, T), where C is the number of channels
        and T is the length of the signal.
    alpha : float
        Parameter defining the bandwidth of extracted modes. A lower value yields a higher bandwidth.
    tau : float

        Time-step of the dual ascent. Use 0 for noise-slack.
    K : int
        Number of modes to be recovered.
    DC : bool
        True if the first mode is put and kept at DC (0-frequency).
    init : int
        Initialization method for omegas:
            - 0: All omegas start at 0
            - 1: All omegas start uniformly distributed
            - 2: All omegas initialized randomly
    tol : float
        Tolerance value for the convergence of ADMM.

    Returns:
    --------
    u : array-like
        A 3D matrix containing K multivariate modes, each with C number of channels and length T.
        Access individual modes or channels using indexing, e.g., u[k, :, c].
    u_hat : array-like
        A 3D matrix containing the spectra of the modes.
    omega : array-like
        A 2D matrix containing the estimated mode center frequencies.

    Notes:
    ------
    The function applies the MVMD algorithm to multivariate or multichannel data sets. It has been verified through
    simulations involving synthetic and real-world data sets containing 2-16 channels.

    Examples:
    ---------
    Example 1: Mode Alignment on Synthetic Data
    >>> T = 1000
    >>> t = np.arange(1, T + 1) / T
    >>> f_channel1 = np.cos(2 * np.pi * 2 * t) + 1 / 16 * np.cos(2 * np.pi * 36 * t)
    >>> f_channel2 = 1 / 4 * np.cos(2 * np.pi * 24 * t) + 1 / 16 * np.cos(2 * np.pi * 36 * t)
    >>> f = np.array([f_channel1, f_channel2])
    >>> u, u_hat, omega = MVMD_new(f, 2000, 0, 3, False, 1, 1e-7)

    Example 2: Real World Data (EEG Data)
    >>> data = np.load('EEG_data.npy')
    >>> u, u_hat, omega = MVMD_new(data, 2000, 0, 6, False, 1, 1e-7)

    Authors: Naveed ur Rehman and Hania Aftab
    Contact Email: naveed.rehman@comsats.edu.pk

    Acknowledgments:
    The MVMD code has been developed by modifying the univariate variational mode decomposition code that has
    been made public at the following link:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    by K. Dragomiretskiy, D. Zosso.

    Please cite the following papers if you use this code in your work:
    [1] N. Rehman, H. Aftab, Multivariate Variational Mode Decomposition, arXiv:1907.04509, 2019.
    [2] K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Transactions on Signal Processing,
        vol. 62, pp. 531-544, 2014.
    """
    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Convert signal to MATLAB compatible type (double)
        signal_matlab = matlab.double(signal.tolist())
        alpha = matlab.double(alpha)
        tau = matlab.double(tau)
        K = matlab.int64(K)
        DC = matlab.logical(DC)
        init = matlab.int64(init)
        tol = matlab.double(tol)

        # Call MATLAB MVMD_new function
        u, u_hat, omega = eng.MVMD_new(signal_matlab, alpha, tau, K, DC, init, tol, nargout=3)

        # Convert MATLAB results to NumPy arrays
        u = np.array(u)
        u_hat = np.array(u_hat)
        omega = np.array(omega)

        return u, u_hat, omega

    finally:
        # Stop MATLAB engine
        if 'eng' in locals():
            eng.quit()


