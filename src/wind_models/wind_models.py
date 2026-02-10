import copy
import time
import warnings
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from shapely import LineString


#######################################################################
# NOTES FOR USER:
# Parameter units listed in []
# Characteristic values listed in {}
#######################################################################


def varying_drag_coefficient_donelan(
    tangential_wind_speed_at_outer_radius: float,
    current_normalized_angular_momentum: float,
    current_normalized_radius: float,
) -> float:
    """
    Donelan et al. (2004) drag coefficient parameterization for tropical cyclones.
    """
    c_d_low_v = 6.2e-4
    v_thresh1 = 6  # m/s transition from constant to linear increasing
    v_thresh2 = 35.4  # m/s transition from linear increasing to constant
    c_d_high_v = 2.35e-3
    slope = (c_d_high_v - c_d_low_v) / (v_thresh2 - v_thresh1)

    # Calculate V at this r/r0 (for variable C_d only)
    v_temp = tangential_wind_speed_at_outer_radius * (
        (current_normalized_angular_momentum / current_normalized_radius)
        - current_normalized_radius
    )

    # Calculate drag_coefficient
    if v_temp <= v_thresh1:
        drag_coefficient = c_d_low_v
    elif v_temp > v_thresh2:
        drag_coefficient = c_d_high_v
    else:
        drag_coefficient = c_d_low_v + slope * (v_temp - v_thresh1)

    return drag_coefficient


def outerwind_r0input_nondim_MM0_E04(
    r0,
    coriolis_factor: float,
    varying_drag_coefficient: bool,
    drag_coefficient: float,
    cooling_rate: float,
    number_radial_divisions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    non-dimensional wind profile outside the radius of maximum wind
    for a tropical cyclone based on the Emanuel (2004) model.

    Returns:
        normalized_radii (numpy.ndarray): Array of non-dimensional radii (r/r0).
        normalized_angular_momentum_array (numpy.ndarray): Array of non-dimensional angular momentum (M/M0).

    Notes:
        - The function assumes absolute values for the Coriolis parameter."""
    coriolis_factor = np.abs(coriolis_factor)

    # [m2/s] M at outer radius
    angular_momentum_at_outer_radius = 0.5 * coriolis_factor * r0 ** 2

    # [m/s] V at outer radius
    tangential_wind_speed_at_outer_radius = angular_momentum_at_outer_radius / r0

    # normalized_step_size = 0.0001
    # grid radii must be > 0
    # if 1 / normalized_step_size < number_radial_divisions:
    #     number_radial_divisions = int(1 / normalized_step_size)
    normalized_step_size = 1. / number_radial_divisions
    assert normalized_step_size < 1.0

    # [-] start at r0, move radially inwards
    max_normalized_radius = 1
    min_normalized_radius = (
        max_normalized_radius - (number_radial_divisions - 1) * normalized_step_size
    )

    # [] r/r0 vector
    normalized_radii = np.arange(
        min_normalized_radius,
        max_normalized_radius + normalized_step_size,
        normalized_step_size,
    )

    # [] M/angular_momentum_at_outer_radius vector initialized to 1 (M/angular_momentum_at_outer_radius = 1 at r/r0=1)
    normalized_angular_momentum_array = np.nan * np.zeros(normalized_radii.shape)
    normalized_angular_momentum_array[-1] = 1

    # First step inwards from r0: d(M/angular_momentum_at_outer_radius)/d(r/r0) = 0 by definition
    current_normalized_radius = normalized_radii[-2]  # one step inwards from r0

    # normalized_momentum_derivative = 0  #[] d(M/angular_momentum_at_outer_radius)/d(r/r0) = 0 at r/r0 = 1
    current_normalized_angular_momentum = normalized_angular_momentum_array[-1]
    normalized_angular_momentum_array[-2] = current_normalized_angular_momentum

    # Integrate inwards from r0 to obtain profile of M/angular_momentum_at_outer_radius vs. r/r0
    for ii in np.arange(normalized_angular_momentum_array.shape[0] - 2, -1, -1):
        if varying_drag_coefficient:
            drag_coefficient = varying_drag_coefficient_donelan(
                tangential_wind_speed_at_outer_radius,
                current_normalized_angular_momentum,
                current_normalized_radius,
            )
        else:
            drag_coefficient = drag_coefficient

        # Calculate [] non-dimensional model parameter
        gamma = drag_coefficient * coriolis_factor * r0 / cooling_rate

        # Update normalized_momentum_derivative at next step inwards
        numerator = (
                        current_normalized_angular_momentum - current_normalized_radius ** 2
                    ) ** 2
        denominator = 1 - current_normalized_radius ** 2
        normalized_momentum_derivative = gamma * numerator / denominator
        normalized_momentum_delta = (
            normalized_momentum_derivative * normalized_step_size
        )

        # Integrate M/angular_momentum_at_outer_radius radially inwards
        current_normalized_angular_momentum = (
            current_normalized_angular_momentum - normalized_momentum_delta
        )

        # Update r/r0 to follow M/angular_momentum_at_outer_radius
        # [] move one step inwards
        current_normalized_radius = current_normalized_radius - normalized_step_size

        # Save updated value
        normalized_angular_momentum_array[ii] = current_normalized_angular_momentum

    return normalized_radii, normalized_angular_momentum_array


def ER11_radprof_raw(
    v_max: float,
    r_max: float,
    rmax_or_r0: str,  # TODO: remove
    coriolis_factor: float,
    CkCd: float,
    radii: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Calculates the Emanuel and Rotunno (2011) theoretical wind speed profile.

    Args:
        v_max (float): Maximum wind speed.
        r_max (float): Input radius, interpreted based on rmax_or_r0.
        rmax_or_r0 (str): Indicates whether r_in is to be treated as rmax ('rmax').
        coriolis_factor (float): Coriolis parameter.
        CkCd (float): Ratio of exchange coefficients.
        radii (np.ndarray): Array of radii at which to calculate the wind speeds.

    Returns:
        Tuple[np.ndarray, list]: Wind speed profile (wind_velocity_er11) and calculated outer radius (r_out).
    """
    coriolis_factor = np.abs(coriolis_factor)

    # Calculate the Emanuel and Rotunno (2011) theoretical profile
    wind_velocity_er11 = (
        np.divide(1.0, radii, where=radii != 0)
        * (v_max * r_max + 0.5 * coriolis_factor * r_max ** 2)
        * ((2 * (radii / r_max) ** 2) / (2 - CkCd + CkCd * (radii / r_max) ** 2))
        ** (1 / (2 - CkCd))
        - 0.5 * coriolis_factor * radii
    )

    # Find the radius corresponding to the outer edge of the wind profile
    max_index = np.argmax(wind_velocity_er11)
    f_ = interp1d(
        wind_velocity_er11[max_index + 1:],
        radii[max_index + 1:],
        fill_value='extrapolate',
    )
    r_out = f_(0.0).tolist()

    return wind_velocity_er11, r_out


def ER11_radprof(
    v_max: float,
    r_in: float,
    rmax_or_r0: str,
    coriolis_factor: float,
    CkCd: float,
    rr_ER11: np.ndarray,
    max_iter: int = 20,
) -> Tuple[np.ndarray, float]:
    """
    Iteratively adjusts and calculates the Emanuel and Rotunno (2011)
    theoretical wind speed profile to fit specified maximum wind speed
    and input radius.

    Args:
        v_max (float): Target maximum wind speed.
        r_in (float): Input radius for adjustment.
        coriolis_factor (float): Coriolis parameter.
        CkCd (float): Ratio of exchange coefficients.
        rr_ER11 (np.ndarray): Array of radii at which to calculate the wind speed

    Returns:
        Tuple[np.ndarray, float]: Adjusted wind speed profile and outer radius.
    """
    coriolis_factor = np.abs(coriolis_factor)
    dr = rr_ER11[1] - rr_ER11[0]

    wind_velocity_er11, r_out = ER11_radprof_raw(
        v_max, r_in, rmax_or_r0, coriolis_factor, CkCd, rr_ER11
    )

    if rmax_or_r0 == 'rmax':
        drin_temp = r_in - rr_ER11[np.argmax(wind_velocity_er11)]
    else:
        assert rmax_or_r0 == 'r0'
        f_ = interp1d(wind_velocity_er11[2:], rr_ER11[2:], fill_value='extrapolate')
        drin_temp = r_in - f_(0).tolist()
    dVmax_temp = v_max - np.max(wind_velocity_er11)

    # Check if errors are too large and adjust accordingly
    n_iter = 0
    r_in_save, v_max_save = copy.deepcopy(r_in), copy.deepcopy(v_max)
    while (np.abs(drin_temp) > dr / 2) or (np.abs(dVmax_temp / v_max_save) >= 10 ** -2):
        # if error is sufficiently large NOTE: FIRST ARGUMENT MUST BE ">" NOT ">=" or else rmax values at exactly dr/2 intervals (e.g. 10.5 for dr=1 km) will not converge
        # drin_temp/1000
        n_iter = n_iter + 1
        if n_iter > max_iter:
            # print('ER11 Params:', v_max, r_in, rmax_or_r0, coriolis_factor, CkCd, rr_ER11)
            # warnings.warn('ER11 CALCULATION DID NOT CONVERGE TO INPUT')
            wind_velocity_er11 = np.nan * np.zeros(rr_ER11.size)
            r_out = np.nan

        # Adjust estimate of r_in according to error
        r_in = r_in + drin_temp

        # Vmax second
        while np.abs(dVmax_temp / v_max) >= 10 ** -2:  # if error is sufficiently large
            # Adjust the estimate of Vmax according to error
            v_max = v_max + dVmax_temp

            [wind_velocity_er11, r_out] = ER11_radprof_raw(
                v_max, r_in, rmax_or_r0, coriolis_factor, CkCd, rr_ER11
            )
            Vmax_prof = np.max(wind_velocity_er11)
            dVmax_temp = v_max_save - Vmax_prof

        [wind_velocity_er11, r_out] = ER11_radprof_raw(
            v_max, r_in, rmax_or_r0, coriolis_factor, CkCd, rr_ER11
        )
        Vmax_prof = np.max(wind_velocity_er11)
        dVmax_temp = v_max_save - Vmax_prof
        if rmax_or_r0 == 'rmax':
            drin_temp = r_in_save - rr_ER11[np.argmax(wind_velocity_er11)]
        else:
            assert rmax_or_r0 == 'r0'
            f_ = interp1d(wind_velocity_er11[2:], rr_ER11[2:], fill_value='extrapolate')
            drin_temp = r_in_save - f_(0).tolist()

    return wind_velocity_er11, r_out


def ER11E04_nondim_r0input(
    v_max,
    r0,
    coriolis_factor,
    Cdvary,
    C_d,
    cooling_rate,
    CkCdvary,
    CkCd,
    eye_adj,
    alpha_eye,
    return_rmax_only=False,
    rmaxr0_min=0.0000,
    rmaxr0_max=1.0,
    rmaxr0_initial_guess=None,
    drmaxr0_thresh = 0.0001,
    timeout=None,
    debugging=False,
    recurision_depth=-1,
):
    debugging = debugging
    recurision_depth = recurision_depth + 1

    # Initialization
    coriolis_factor = np.abs(coriolis_factor)

    if (CkCdvary == 1) or (CkCdvary is True):
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * v_max ** 2 + CkCd_coeflin * v_max + CkCd_coefcnst
    if CkCd > 1.9:
        warnings.warn(
            'Vmax is much greater than the range of data used to estimate CkCd as a function of Vmax -- here be dragons!'
        )
        CkCd = 1.9

    # Step 1: Calculate E04 M/M0 vs. r/r0
    Nr = 100_000
    rrfracr0_E04, MMfracM0_E04 = outerwind_r0input_nondim_MM0_E04(
        r0, coriolis_factor, Cdvary, C_d, cooling_rate, Nr
    )
    M0_E04 = 0.5 * coriolis_factor * r0 ** 2

    # Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point with E04 M/M0
    count = 0
    soln_converged = 0

    if timeout is not None:
        start_time = time.time()

    while soln_converged == 0:
        count += 1

        # Break up the interval into 3 points, take 2 between which intersection vanishes, repeat til converges
        # rmaxr0_min = 0.0001
        # rmaxr0_max = 1.0
        # rmaxr0_new_guess = (rmaxr0_max + rmaxr0_min) / 2.0  # first guess -- in the middle
        rmaxr0_min = rmaxr0_min
        rmaxr0_max = rmaxr0_max

        rmaxr0_new_guess = (rmaxr0_max + rmaxr0_min) / 2.0
        if rmaxr0_initial_guess is not None and count == 1:
            rmaxr0_new_guess = float(rmaxr0_initial_guess)

        rmaxr0_previous_guess = rmaxr0_new_guess  # initialize
        # drmaxr0 = rmaxr0_max - rmaxr0_previous_guess  # initialize
        drmaxr0 = (rmaxr0_max - rmaxr0_min) / 2.0


        # Threshold for derivative difference
        # drmaxr0_thresh = 0.0001
        drmaxr0_thresh = drmaxr0_thresh
        # drmaxr0_thresh = 500 / r0

        iterN = 0
        rfracrm_min = 0.0  # [-] start at r=0
        # rfracrm_max = 500.0  # [-] extend out to many rmaxs
        rfracrm_max = 50.0  # [-] extend out to many rmaxs
        while np.abs(drmaxr0) >= drmaxr0_thresh:
            iterN = iterN + 1
            # Calculate ER11 M/Mm vs r/rm
            rmax = rmaxr0_new_guess * r0  # [m]r
            #     [~,~,rrfracrm_ER11,MMfracMm_ER11] =
            #     ER11_radprof_nondim(Vmax,rmax,fcor,CkCd) #FAILS FOR LOW CK/CD NOT SURE WHY

            # drfracrm = 0.01
            # if rmax > 100.0 * 1000:
            #     drfracrm = drfracrm / 10.0  # extra precision for large storm
            # TODO: Altered to speed things up
            drfracrm = 0.01


            # [] r/r0 vector
            rrfracrm_ER11 = np.arange(
                rfracrm_min, rfracrm_max + drfracrm, drfracrm
            )
            rr_ER11 = rrfracrm_ER11 * rmax
            rmax_or_r0 = 'rmax'

            try:
                # print(v_max)
                VV_ER11, dummy = ER11_radprof(
                    v_max, rmax, rmax_or_r0, coriolis_factor, CkCd, rr_ER11
                )

            except:
                VV_ER11 = np.nan * np.zeros(rr_ER11.size)
            if not np.isnan(np.max(VV_ER11)):  # ER11_radprof converged
                rrfracr0_ER11 = rr_ER11 / r0
                MMfracM0_ER11 = (
                                    rr_ER11 * VV_ER11 + (0.5 * coriolis_factor * rr_ER11 ** 2)
                                ) / M0_E04

                l1 = LineString(zip(rrfracr0_E04, MMfracM0_E04))
                l2 = LineString(zip(rrfracr0_ER11, MMfracM0_ER11))
                intersection = l1.intersection(l2)
                if (
                    'EMPTY' in intersection.wkt
                ):
                    # no intersections -- rmaxr0 too small
                    # drmaxr0 = np.abs(drmaxr0) / 2
                    rmaxr0_previous_guess = rmaxr0_new_guess
                    rmaxr0_min = rmaxr0_previous_guess
                    rmaxr0_max = rmaxr0_max
                    rmaxr0_new_guess = (rmaxr0_max + rmaxr0_min) / 2.0
                    drmaxr0 = (rmaxr0_max - rmaxr0_min) / 2.0
                    if debugging:
                        print('No intersection')
                else:

                    # at least one intersection -- rmaxr0 too large
                    # drmaxr0 = - np.abs(drmaxr0) / 2
                    rmaxr0_previous_guess = rmaxr0_new_guess
                    rmaxr0_min = rmaxr0_min
                    rmaxr0_max = rmaxr0_previous_guess
                    rmaxr0_new_guess = (rmaxr0_max + rmaxr0_min) / 2.0
                    drmaxr0 = (rmaxr0_max - rmaxr0_min) / 2.0


                    if intersection.wkt.split(' ')[0] == 'POINT':
                        X0, Y0 = intersection.coords[0]
                        # then we are very close to the solution or way too big
                        if debugging:
                            print('Points:', X0, Y0)
                    elif intersection.wkt.split(' ')[0] == 'MULTIPOINT':
                        X0, Y0 = intersection.geoms[0].coords[0]
                        X1, Y1 = intersection.geoms[-1].coords[0]
                        if debugging:
                            print('Points:', X0, Y0, X1, Y1)

                    rmerger0 = np.mean(X0)
                    MmergeM0 = np.mean(Y0)


            else:  # ER11_radprof did not converge -- convergence fails for low CkCd and high Ro = Vm/(f*rm)
                ## Must reduce rmax (and thus reduce Ro)
                # drmaxr0 = -abs(drmaxr0) / 2
                rmaxr0_previous_guess = rmaxr0_new_guess
                rmaxr0_min = rmaxr0_min
                rmaxr0_max = rmaxr0_previous_guess
                rmaxr0_new_guess = (rmaxr0_max + rmaxr0_min) / 2.0
                drmaxr0 = (rmaxr0_max - rmaxr0_min) / 2.0


            ## update value of rmaxr0
            # rmaxr0_previous_guess = rmaxr0_new_guess  # this is the final one
            # rmaxr0_new_guess = rmaxr0_new_guess + drmaxr0

            #####################
            if debugging:
                rrfracr0_ER11 = rr_ER11 / r0
                MMfracM0_ER11 = (
                                    rr_ER11 * VV_ER11 + (0.5 * coriolis_factor * rr_ER11 ** 2)
                                ) / M0_E04
                M0 = 0.5 * coriolis_factor * r0 ** 2

                # Compute the values for plotting
                rr_ = rr_ER11
                VV_ = (M0 / r0) * ((MMfracM0_ER11 / rrfracr0_ER11) - rrfracr0_ER11)
                rr_E04 = rrfracr0_E04 * r0
                VV_E04 = (M0 / r0) * ((MMfracM0_E04 / rrfracr0_E04) - rrfracr0_E04)

                # Derivatives for the second plot
                derivative_ER11 = np.gradient(VV_, rr_)
                derivative_E04 = np.gradient(VV_E04, rr_E04)

                # Setup figure and axes
                # fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(12, 5))
                #
                # # Plotting the original data
                # ax1.plot(rr_ / 1000, VV_, 'b', linewidth=2, label='ER11')
                # ax1.plot(rr_E04 / 1000, VV_E04, 'r', linewidth=2, label='E04')
                # ax1.set_xlim(0, r0 * 1.25 / 1000)
                # ax1.set_ylim(0, v_max * 1.75)
                # ax1.set_xlabel('r [km]')
                # ax1.set_ylabel('V [m/s]')
                # ax1.set_title('Wind Speed Profile')
                # ax1.legend()
                #
                # # Plotting the derivatives
                # ax2.plot(rr_ / 1000, derivative_ER11 * -1, 'b', linewidth=2, label='Derivative ER11')
                # ax2.plot(rr_E04 / 1000, derivative_E04 * -1, 'r', linewidth=2, label='Derivative E04')
                # ax2.set_xlim(0, r0 * 1.25 / 1000)
                # # ax2.set_ylim(-0.1, 0.5)
                # ax2.set_yscale('log')
                # ax2.set_xlabel('r [km]')
                # ax2.set_ylabel('dV/dr [m/s per km]')
                # ax2.set_title('Derivative of Wind Speed')
                # ax2.legend()
                #
                # plt.tight_layout()
                # plt.show()
                fig, ax1 = plt.subplots(dpi=100, figsize=(5, 5))

                # Plotting the original data
                ax1.plot(rr_ / 1000, VV_, 'b', linewidth=2, label='ER11')
                ax1.plot(rr_E04 / 1000, VV_E04, 'r', linewidth=2, label='E04')
                ax1.set_xlim(0, r0 * 1.25 / 1000)
                ax1.set_ylim(0, v_max * 1.75)
                ax1.set_xlabel('r [km]')
                ax1.set_ylabel('V [m/s]')
                ax1.set_title('Wind Speed Profile')
                ax1.legend()

                plt.tight_layout()
                plt.show()



        ## Check if solution converged
        if (not np.isnan(np.max(VV_ER11))) and ('rmerger0' in locals()):
            soln_converged = 1
        else:
            soln_converged = 0
            CkCd = CkCd + 0.1
            if debugging:
                print(f'Adjusting CkCd to find convergence, CkCd: {CkCd}')

            if CkCd > 1.9:
                if recurision_depth > 2:
                    print('FAILED')
                    print(f'Timed Out. V_max: {v_max}, r0: {r0}, coriolis_factor: {coriolis_factor}', end=' ')
                    print(f'Cdvary: {Cdvary}, C_d: {C_d}, cooling_rate: {cooling_rate}', end=' ')
                    print(f'CkCdvary: {CkCdvary}, CkCd: {CkCd}, eye_adj: {eye_adj}, alpha_eye: {alpha_eye}')
                    print(f'Rmaxr0: {rmaxr0_new_guess}, Iterations: {iterN}')
                    raise TimeoutError('Failed to converge')

                # just restart the fit again
                # call the function again
                return ER11E04_nondim_r0input(
                    v_max,
                    r0,
                    coriolis_factor,
                    Cdvary=1,
                    C_d=1.5e-3,
                    cooling_rate=2e-3,
                    CkCdvary=0,
                    CkCd=1,
                    eye_adj=0,
                    alpha_eye=0.15,
                    return_rmax_only=return_rmax_only,
                    rmaxr0_min=0.000,
                    rmaxr0_max=0.999,
                    rmaxr0_initial_guess=rmaxr0_new_guess,
                    timeout=timeout,
                    debugging=debugging,
                    recurision_depth=recurision_depth,
                )


        if timeout is not None and soln_converged == 0:
            if time.time() - start_time > timeout:
                # print(f'Timed Out. V_max: {v_max}, r0: {r0}, coriolis_factor: {coriolis_factor}', end=' ')
                # print(f'Cdvary: {Cdvary}, C_d: {C_d}, cooling_rate: {cooling_rate}', end=' ')
                # print(f'CkCdvary: {CkCdvary}, CkCd: {CkCd}, eye_adj: {eye_adj}, alpha_eye: {alpha_eye}')
                # print(f'Rmaxr0: {rmaxr0_new_guess}, Iterations: {iterN}')
                raise TimeoutError

            # print('Adjusting CkCd to find convergence')
    # print(f'Converged after {iterN} iterations', iterN)
    if return_rmax_only:
        return rmax

    # Calculate some things
    M0 = 0.5 * coriolis_factor * r0 ** 2
    Mm = 0.5 * coriolis_factor * rmax ** 2 + rmax * v_max
    MmM0 = Mm / M0

    # Finally: Interpolate to a grid
    ii_ER11 = np.argwhere((rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0))[:, 0]
    ii_E04 = np.argwhere((rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0))[:, 0]

    MMfracM0_temp = np.hstack((MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04]))
    rrfracr0_temp = np.hstack((rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04]))
    del ii_ER11
    del ii_E04

    #### PLOT SOME THINGS (DEBUGGING)
    # rrfracr0_E04, MMfracM0_E04
    rr_ = rr = rrfracr0_E04 * r0
    VV_ = (M0 / r0) * ((MMfracM0_E04 / rrfracr0_E04) - rrfracr0_E04)
    # fig, ax = plt.subplots(dpi=100)
    # ax.plot(rr_ / 1000, VV_, 'b', linewidth=2, label='E04')
    # ax.set_xlim(0, 1050)
    # ax.set_ylim(0, 55)
    # ax.set_xlabel('r [km]')
    # ax.set_ylabel('V [m/s]')
    # # ax.set_title('E04')
    # # plt.show()
    #
    # # rrfracr0_ER11, MMfracM0_ER11
    # rr_ = rr_ER11
    # VV_ = (M0 / r0) * ((MMfracM0_ER11 / rrfracr0_ER11) - rrfracr0_ER11)
    # # fig, ax = plt.subplots(dpi=100)
    # ax.plot(rr_ / 1000, VV_, 'r', linewidth=2, label='ER11')
    # ax.set_xlim(0, 1050)
    # ax.set_ylim(0, 55)
    # ax.set_xlabel('r [km]')
    # ax.set_ylabel('V [m/s]')
    # # ax.set_title('ER11')
    # plt.show()

    # calculating VV at radii relative to rmax ensures no smoothing near rmax!
    drfracrm = (
        0.01
    )
    rfracrm_min = 0  # [-] r=0
    rfracrm_max = r0 / rmax  # [-] r=r0
    rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)  # [] r/r0 vector
    f = interp1d(rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm), fill_value='extrapolate')
    MMfracMm = f(rrfracrm)

    rrfracr0 = rrfracrm * rmax / r0  # save this as output
    MMfracM0 = MMfracMm * Mm / M0

    # Calculate dimensional wind speed and radii
    VV = (Mm / rmax) * (
        MMfracMm / rrfracrm
    ) - (0.5 * coriolis_factor * rmax * rrfracrm)  # [ms-1]
    rr = rrfracrm * rmax  # [m]

    # Make sure V=0 at r=0
    VV[rr == 0] = 0

    rmerge = rmerger0 * r0
    Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [ms-1]

    # Adjust profile in eye, if desired
    # if(eye_adj==1)
    # 	r_eye_outer = rmax
    # 	V_eye_outer = Vmax
    # 	[VV] = radprof_eyeadj(rr,VV,alpha_eye,r_eye_outer,V_eye_outer)
    #
    #    print('EYE ADJUSTMENT: eye alpha = #3.2f',alpha_eye)

    if debugging:
        fig, ax1 = plt.subplots(dpi=100, figsize=(5, 5))

        # Plotting the original data
        ax1.plot(rr_ / 1000, VV_, 'b', linewidth=2, label='ER11')
        ax1.plot(rr_E04 / 1000, VV_E04, 'r', linewidth=2, label='E04')
        ax1.plot(rr / 1000, VV, 'xkcd:purple', linewidth=2, label='Final Fit')
        ax1.set_xlim(0, r0 * 1.25 / 1000)
        ax1.set_ylim(0, v_max * 1.75)
        ax1.set_xlabel('r [km]')
        ax1.set_ylabel('V [m/s]')
        ax1.set_title('Wind Speed Profile')
        ax1.legend()

        plt.tight_layout()
        plt.show()

    return rr, VV, rmerge, Vmerge, rmax


def ER11E04_nondim_rmaxinput_FAST_XGBOOST(
    Vmax,
    rmax,
    fcor,
    model=None,
    *args,
    **kwargs,
):
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model('model.json')
    model.predict([[Vmax, rmax, fcor]])


def ER11E04_nondim_rmaxinput(Vmax, rmax, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye):
    ## Initialization
    fcor = np.abs(fcor)
    if CkCdvary == 1:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax ** 2 + CkCd_coeflin * Vmax + CkCd_coefcnst
    ## 'Ck/Cd is capped at 1.9 and has been set to this value. If CkCdvary=1,
    ## then Vmax is much greater than the range of data used to estimate
    ## CkCd as a function of Vmax -- here be dragons!')
    CkCd = np.min((1.9, CkCd))
    ## Step 1: Calculate ER11 M/Mm vs. r/rm
    # [~,~,rrfracrm_ER11,MMfracMm_ER11] = ER11_radprof_nondim(Vmax,rmax,fcor,CkCdvary,CkCd)
    drfracrm = .01
    if (rmax > 100. * 1000):
        drfracrm = drfracrm / 10.  # extra precision for large storm


    rfracrm_min = 0.  # [-] start at r=0
    rfracrm_max = 25.  # [-] extend out to many rmaxs
    rrfracrm_ER11 = np.arange(rfracrm_min, rfracrm_max + drfracrm, drfracrm)  # [] r/r0 vector
    rr_ER11 = rrfracrm_ER11 * rmax
    rmax_or_r0 = 'rmax'
    soln_converged = 0
    count = 0
    while soln_converged == 0:
        count += 1
        VV_ER11, dummy = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)
        ## Check if solution converged
        if (not np.isnan(np.max(VV_ER11))):
            soln_converged = 1
        else:
            soln_converged = 0
            CkCd = CkCd + .1
            if rmax == 0.0:
                rmax = 25.
            # print('Adjusting CkCd to find convergence')
            if count >= 50:
                break;
    if soln_converged == 1:
        Mm = .5 * fcor * rmax ** 2 + rmax * Vmax
        MMfracMm_ER11 = (rr_ER11 * VV_ER11 + .5 * fcor * rr_ER11 ** 2) / Mm
        ## Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point with E04 M/M0
        ##Break up interval into 3 points, take 2 between which intersection vanishes, repeat til converges
        rmaxr0_min = .001
        rmaxr0_max = .75
        # rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2  # first guess -- in the middle

        # rmaxr0_min = 0.05
        # rmaxr0_max = 0.6
        rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2

        rmaxr0 = rmaxr0_new  # initialize
        drmaxr0 = rmaxr0_max - rmaxr0  # initialize
        drmaxr0_thresh = .000001
        iterN = 0
        while np.abs(drmaxr0) >= drmaxr0_thresh:
            iterN = iterN + 1
            ##Calculate E04 M/M0 vs r/r0
            r0 = rmax / rmaxr0_new  # [m]
            Nr = 100000
            rrfracr0_E04, MMfracM0_E04 = outerwind_r0input_nondim_MM0_E04(r0, fcor, Cdvary, C_d, w_cool, Nr)
            ##Convert ER11 to M/M0 vs. r/r0 space
            rrfracr0_ER11 = rrfracrm_ER11 * (rmaxr0_new)
            M0_E04 = .5 * fcor * r0 ** 2
            MMfracM0_ER11 = MMfracMm_ER11 * (Mm / M0_E04)
            l1 = LineString(zip(rrfracr0_E04, MMfracM0_E04))
            l2 = LineString(zip(rrfracr0_ER11, MMfracM0_ER11))
            intersection = l1.intersection(l2)
            if 'EMPTY' in intersection.wkt:  ##no intersections r0 too large --> rmaxr0 too small
                drmaxr0 = np.abs(drmaxr0) / 2
            else:  ##at least one intersection -- r0 too small --> rmaxr0 too large
                if intersection.wkt.split(' ')[0] == 'POINT':
                    X0, Y0 = intersection.coords[0]
                elif intersection.wkt.split(' ')[0] == 'MULTIPOINT':
                    X0, Y0 = intersection.geoms[0].coords[0]
                # at least one intersection -- rmaxr0 too large
                drmaxr0 = -np.abs(drmaxr0) / 2
                rmerger0 = np.mean(X0)
                MmergeM0 = np.mean(Y0)
            ###update value of rmaxr0
            rmaxr0 = rmaxr0_new  # this is the final one
            rmaxr0_new = rmaxr0_new + drmaxr0
        if 'EMPTY' in intersection.wkt:
            ## Calculate some things
            M0 = .5 * fcor * r0 ** 2
            Mm = .5 * fcor * rmax ** 2 + rmax * Vmax
            MmM0 = Mm / M0

            ## Finally: Interpolate to a grid
            ## Finally: Interpolate to a grid
            ii_ER11 = np.argwhere((rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0))[:, 0]
            ii_E04 = np.argwhere((rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0))[:, 0]
            MMfracM0_temp = np.hstack((MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04]))
            rrfracr0_temp = np.hstack((rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04]))
            del ii_ER11
            del ii_E04
            rfracrm_min = 0  # [-] r=0
            rfracrm_max = r0 / rmax  # [-] r=r0
            rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)  # [] r/r0 vector
            f = interp1d(rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm), fill_value='extrapolate')
            MMfracMm = f(rrfracrm)

            rrfracr0 = rrfracrm * rmax / r0  # save this as output
            MMfracM0 = MMfracMm * Mm / M0

            ## Calculate dimensional wind speed and radii
            # VV = (M0/r0)*((MMfracM0./rrfracr0)-rrfracr0)  #[ms-1]
            # rr = rrfracr0*r0   #[m]
            # rmerge = rmerger0*r0
            # Vmerge = (M0/r0)*((MmergeM0./rmerger0)-rmerger0)  #[ms-1]
            VV = (Mm / rmax) * (MMfracMm / rrfracrm) - .5 * fcor * rmax * rrfracrm  # [ms-1]
            rr = rrfracrm * rmax  # [m]

            ## Make sure V=0 at r=0
            VV[rr == 0] = 0

            rmerge = rmerger0 * r0
            Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [ms-1]

            ## Adjust profile in eye, if desired
        # if(eye_adj==1)
        #	r_eye_outer = rmax
        #	V_eye_outer = Vmax
        #	[VV] = radprof_eyeadj(rr,VV,alpha_eye,r_eye_outer,V_eye_outer)
        #    sprintf('EYE ADJUSTMENT: eye alpha = #3.2f',alpha_eye)
        else:
            rr = rr_ER11
            VV = VV_ER11
            rmerge = np.nan
            Vmerge = np.nan
    else:
        rr = np.nan * np.zeros(10)
        VV = np.nan * np.zeros(10)
        r0 = np.nan
        rmerge = np.nan
        Vmerge = np.nan
    # print(rmax)
    return rr, VV, r0, rmerge, Vmerge



if __name__ == '__main__':
    from tqdm import tqdm

    samples = []

    for i in tqdm(range(0, 100)):
        warnings.filterwarnings('ignore')
        result = ER11E04_nondim_r0input(
            v_max=25.39900638154602,
            r0=623610.5,
            coriolis_factor=7.748082134639844e-05,
            Cdvary=1,
            C_d=1.5e-3,
            cooling_rate=2e-3,
            CkCdvary=0,
            CkCd=1,
            eye_adj=0,
            alpha_eye=0.15,
            return_rmax_only=True,
            # rmaxr0_min=0.0001,
            # rmaxr0_max=1.0,
            rmaxr0_min=0.0,
            rmaxr0_max=1.0,
            debugging=False,
            timeout=3
        )
        result = result / 623610.5
        samples.append(result)
    samples = np.array(samples)
