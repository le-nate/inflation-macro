import numpy as np
import matplotlib.pyplot as plt

# import pycwt as wavelet
import pycwt as wavelet

from simulation_consumption import consumption


def main() -> None:
    """Run script"""
    t = np.linspace(0, 2 * np.pi, 1000)
    dt = t[1] - t[0]
    dj = 1 / 12
    signal1 = np.sin(2 * np.pi * 7 * t) + np.random.lognormal(0, 0.1, 1000)

    signal2 = np.sin(2 * np.pi * 5 * t) + np.random.lognormal(0, 0.1, 1000)

    mother = wavelet.Morlet(6)  # Morlet wavelet with :math:`\omega_0=6`.

    xwt_result, coi, freqs, signif = wavelet.xwt(
        signal1, signal2, dt=dt, dj=dj, wavelet=mother
    )

    print(xwt_result)
    print(coi)
    print(freqs)
    print(signif)

    # * Plot results
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(xwt_result),
        extent=[-1, 1, 1, 31],
        cmap="jet",
        aspect="auto",
        vmax=np.abs(xwt_result).max(),
        vmin=-np.abs(xwt_result).max(),
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Continuous Wavelet Transform")
    plt.show()
    # plt.close("all")
    # # plt.ioff()
    # figprops = {"figsize": (11, 8), "dpi": 72}
    # fig = plt.figure(**figprops)

    # # 1) original series anomaly and the inverse wavelet transform
    # ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    # ax.plot(t, signal1, "-", linewidth=1, color=[0.5, 0.5, 0.5])
    # ax.plot(t, signal2, "k", linewidth=1.5)

    # # * Normalized cwt power spectrum, signifance levels, and cone of influence
    # # Period scale is logarithmic
    # bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    # bx.contourf(
    #     t,
    #     np.log2(period),
    #     np.log2(power),
    #     np.log2(levels),
    #     extend="both",
    #     cmap=plt.cm.jet,
    # )
    # extent = [t.min(), t.max(), 0, max(period)]
    # bx.contour(
    #     t, np.log2(period), sig95, [-99, 1], colors="k", linewidths=2, extent=extent
    # )
    # bx.fill(
    #     np.concatenate([t, t[-1:] + DT, t[-1:] + DT, t[:1] - DT, t[:1] - DT]),
    #     np.concatenate(
    #         [
    #             np.log2(coi),
    #             [1e-9],
    #             np.log2(period[-1:]),
    #             np.log2(period[-1:]),
    #             [1e-9],
    #         ]
    #     ),
    #     "k",
    #     alpha=0.3,
    #     hatch="x",
    # )
    # bx.set_title("b) {} Wavelet Power Spectrum ({})".format(LABEL, mother.name))
    # bx.set_ylabel("Period (years)")
    # #
    # Yticks = 2 ** np.arange(
    #     np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))
    # )
    # bx.set_yticks(np.log2(Yticks))
    # bx.set_yticklabels(Yticks)

    # plt.show()


if __name__ == "__main__":
    main()
