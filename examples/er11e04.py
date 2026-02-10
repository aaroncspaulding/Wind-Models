from wind_models import ER11E04_nondim_r0input

import matplotlib.pyplot as plt


def main(show: bool = True):
    result = ER11E04_nondim_r0input(
        # 9.920024138687134, r0: 1198489.75, coriolis_factor: 9.528460213914514e-05
        v_max=50,
        r0=1_200_000,
        # v_max=50,
        # r0=1_000_000,
        coriolis_factor=9.528460213914514e-05,
        Cdvary=1,
        C_d=1.5e-3,
        cooling_rate=2e-3,
        CkCdvary=0,
        CkCd=1,
        eye_adj=0,
        alpha_eye=0.15,
        return_rmax_only=False,
        # rmaxr0_min=0.0001,
        # rmaxr0_max=1.0,
        rmaxr0_min=0.0,
        rmaxr0_max=1.3,
        debugging=False,
        timeout=3,
        drmaxr0_thresh=0.00001,
    )
    rr, VV, rmerge, Vmerge, rmax = result

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(rr / 1000, VV, "b", linewidth=2, label="Nondim M soln")
    ax.set_xlabel("r [km]")
    ax.set_ylabel("V [m/s]")
    ax.legend()

    if show:
        plt.show()
    else:
        plt.close("all")

    return rr, VV, rmerge, Vmerge, rmax


if __name__ == "__main__":
    main(show=True)
