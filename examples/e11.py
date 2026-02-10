import matplotlib.pyplot as plt
import numpy as np

from wind_models import ER11_radprof_raw


def main(show: bool = True):
    # User input parameters
    vmax = 50  # m/s
    rmax_or_r0 = "rmax"
    r_in = 40 * 1000  # m
    fcor = 5e-5
    ckcd = 1.9

    radii = np.arange(0, 1000 * 1000, 1000)  # m

    v_er11, r_out = ER11_radprof_raw(vmax, r_in, rmax_or_r0, fcor, ckcd, radii)
    v_er11 = np.clip(v_er11, 0, 100)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(radii / 1000, v_er11, "b", linewidth=2, label="Nondim M soln")
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 100)
    ax.set_xlabel("r [km]")
    ax.set_ylabel("V [m/s]")

    if show:
        plt.show()
    else:
        plt.close("all")

    return radii, v_er11, r_out


if __name__ == "__main__":
    main(show=True)
