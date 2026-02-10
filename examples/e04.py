import matplotlib.pyplot as plt

from chavas_2015.wind_models import outerwind_r0input_nondim_MM0_E04

# User input parameters
r0 = 2133 * 1000  # [m]
fcor = 5e-5  # [s^-1]
Cdvary = False
C_d = 1.5e-3  # [-]
w_cool = 2e-3  # [m/s]
Nr = 10000  # [-]; set to a large number to integrate all the way to near 0
V_max = 100  # [m/s]



# Calculate non-dimensional model parameter (gamma)
gam = C_d * fcor * r0 / w_cool

# Calculate non-dimensional solution
rrfracr0, MMfracM0 = outerwind_r0input_nondim_MM0_E04(r0, fcor, Cdvary, C_d, w_cool, Nr)

coriolis_factor = fcor
M0 = 0.5 * coriolis_factor * r0**2
v = (M0 * MMfracM0 / rrfracr0) - (coriolis_factor * rrfracr0 / 2)

VV = (M0 / r0) * ((MMfracM0 / rrfracr0) - rrfracr0)
rr = rrfracr0 * r0




fig, ax = plt.subplots(dpi=100)
ax.plot(rrfracr0, MMfracM0, 'b', linewidth=2, label='Nondim M soln')
ax.plot(1, 1, 'r*', markersize=14, linewidth=2)
ax.set_xlabel('r/r_0')
ax.set_ylabel('M/M_0')
ax.set_title(f'gam = {gam:3.1f}')
ax.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(dpi=100)
ax.plot(rr / 1000, VV, 'b', linewidth=2)
ax.set_xlim(0, 2500)
ax.set_ylim(0, 100)
ax.set_xlabel('r [km]')
ax.set_ylabel('V [m/s]')
plt.show()
