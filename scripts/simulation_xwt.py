import numpy as np
import matplotlib.pyplot as plt

# import pycwt as wavelet
from scipy.signal import cwt, morlet

from simulation_consumption import consumption


def main() -> None:
    """Run script"""
    t = np.linspace(0, 2 * np.pi, 1000)
    signal1 = np.sin(2 * np.pi * 7 * t) + np.random.lognormal(0, 0.1, 1000)

    signal2 = np.sin(2 * np.pi * 5 * t) + np.random.lognormal(0, 0.1, 1000)

    # Define wavelet parameters
    widths = np.arange(1, 31)
    wavelet = morlet

    # Perform continuous wavelet transform (CWT) for each time series dataset
    cwt_result1 = cwt(signal1, wavelet, widths)
    cwt_result2 = cwt(signal2, wavelet, widths)

    # Compute cross wavelet transform by element-wise multiplication of CWT results
    cross_wavelet_transform = cwt_result1 * cwt_result2

    # Plot the cross wavelet transform
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(cross_wavelet_transform),
        extent=[0, 2 * np.pi, 1, 30],
        cmap="jet",
        aspect="auto",
        vmax=np.abs(cross_wavelet_transform).max(),
        vmin=0,
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Cross Wavelet Transform")
    plt.show()


if __name__ == "__main__":
    main()
