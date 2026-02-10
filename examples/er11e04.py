from chavas_2015.wind_models import ER11E04_nondim_r0input, ER11E04_nondim_rmaxinput
import numpy as np

import matplotlib.pyplot as plt

Vmax = 50
r0 = 1000 * 1000

fcor = 5e-5
Cdvary = 1
C_d = 1.5e-3
w_cool = 2e-3
CkCdvary = 0
CkCd = 1
eye_adj = 0
alpha_eye = 0.15

# print(my_divide(np.array([1, 1]),
# np.array([0, 2])
#                 ))
rr, VV, rmerge, Vmerge, rmax = ER11E04_nondim_r0input(Vmax,
                                                      r0,
                                                      fcor,
                                                      Cdvary,
                                                      C_d,
                                                      w_cool,
                                                      CkCdvary,
                                                      CkCd,
                                                      eye_adj,
                                                      alpha_eye)

# print('RMAX:', rmax)

# rr, VV, r0, rmerge, Vmerge = ER11E04_nondim_rmaxinput(Vmax,
#                                                       rmax,
#                                                       fcor,
#                                                       Cdvary,
#                                                       C_d,
#                                                       w_cool,
#                                                       CkCdvary,
#                                                       CkCd,
#                                                       eye_adj,
#                                                       alpha_eye)
print(r0)




fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

rr = rr / 1000
ax.plot(rr, VV, 'b', linewidth=2, label='Nondim M soln')

plt.show()
