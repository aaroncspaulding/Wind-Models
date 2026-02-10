import matplotlib.pyplot as plt
import numpy as np

from chavas_2015.wind_models import ER11_radprof_raw, ER11_radprof

# User input parameters
Vmax = 50  #m / s
rmax_or_r0 = 'rmax'
r_in = 40 * 1000 # m
fcor = 5e-5
CkCd = 1.9


radii = np.arange(0, 1000 * 1000, 1000) # m

V_ER11, r_out = ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, radii)
V_ER11 = np.clip(V_ER11, 0, 100)


fig, ax = plt.subplots(dpi=100)
ax.plot(radii / 1000, V_ER11, 'b', linewidth=2, label='Nondim M soln')
ax.set_xlim(0, 2500)
ax.set_ylim(0, 100)
ax.set_xlabel('r [km]')
ax.set_ylabel('V [m/s]')
plt.show()
