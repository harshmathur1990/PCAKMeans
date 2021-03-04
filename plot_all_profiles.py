import h5py
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

base_input_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/inversions/'
)

second_path = base_input_path / 'plots_v1'
profile_files = [
    second_path / 'frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_profs.nc',
]

atmos_files = [
    second_path / 'frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_atmos.nc',
]
observed_file = Path(
    base_input_path / 'frame_0_21_x_662_712_y_708_758.nc'
)


profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')

indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]

write_path = second_path / 'profile_fits'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

size = plt.rcParams['lines.markersize']

fontP = FontProperties()
fontP.set_size('xx-small')

calib_velocity = 333390.00079943583

plt.close('all')
plt.clf()
plt.cla()

k = 0
i = 0
j = 0
fig, axs = plt.subplots(2, 2)

obp,  = axs[0][0].plot(
    observed['wav'][indices],
    observed['profiles'][k, i, j, :, 0][indices],
    color=red,
    linewidth=0.5,
    label='Shock'
)

obs = axs[0][0].scatter(
    observed['wav'][indices],
    observed['profiles'][k, i, j, :, 0][indices],
    color=red,
    s=size / 4
    # linewidth=0.5
    # label='Shock'
)

# plotting the inverted profile
inp, = axs[0][0].plot(
    observed['wav'][:-1],
    profiles[0]['profiles'][k, i, j, :-1, 0],
    color=green,
    linewidth=0.5,
    label='Fit'
)

ins = axs[0][0].scatter(
    observed['wav'][:-1],
    profiles[0]['profiles'][k, i, j, :-1, 0],
    color=green,
    # linewidth=0.5,
    s=size / 4
    # label='Fit'
)

axs[0][0].set_ylim(0, 0.5)

# plot inverted temperature profile
temp, = axs[0][1].plot(
    atmos[0]['ltau500'][k][i][j],
    atmos[0]['temp'][k][i][j],
    color=green,
    linewidth=0.5
)

axs[0][1].set_ylim(4000, 11000)

# plot inverted Vlos profile
vlos, = axs[1][0].plot(
    atmos[0]['ltau500'][k][i][j],
    (atmos[0]['vlos'][k][i][j] - calib_velocity)/ 1e5,
    color=green,
    linewidth=0.5
)

axs[1][0].set_ylim(-20, 20)

# plot inverted Vturb profile
vturb, = axs[1][1].plot(
    atmos[0]['ltau500'][k][i][j],
    atmos[0]['vturb'][k][i][j] / 1e5,
    color=green,
    linewidth=0.5
)

axs[1][1].set_ylim(-10, 10)

fig.tight_layout()

axs[0][0].legend(loc='upper right', prop=fontP)
# plt.show()
plt.savefig(write_path / 'plot_{}_{}_{}.png'.format(k, i + 662, j + 708), format='png', dpi=150)


for k in range(21):
    for i in range(50):
        for j in range(50):

            print ('k={}, i={}, j={}'.format(k, i, j))
            obp.set_ydata(observed['profiles'][k, i, j, :, 0][indices])
            obs.set_offsets(np.c_[observed['wav'][indices], observed['profiles'][k, i, j, :, 0][indices]])
            inp.set_ydata(profiles[0]['profiles'][k, i, j, :-1, 0])
            ins.set_offsets(np.c_[observed['wav'][:-1], profiles[0]['profiles'][k, i, j, :-1, 0]])
            temp.set_ydata(atmos[0]['temp'][k][i][j])
            vlos.set_ydata((atmos[0]['vlos'][k][i][j] - calib_velocity)/ 1e5)
            vturb.set_ydata(atmos[0]['vturb'][k][i][j] / 1e5)
            axs[0][0].draw_artist(axs[0][0].patch)
            axs[0][1].draw_artist(axs[0][1].patch)
            axs[1][0].draw_artist(axs[1][0].patch)
            axs[1][1].draw_artist(axs[1][1].patch)
            axs[0][0].draw_artist(obp)
            axs[0][0].draw_artist(obs)
            axs[0][0].draw_artist(inp)
            axs[0][0].draw_artist(ins)
            axs[0][1].draw_artist(temp)
            axs[1][0].draw_artist(vlos)
            axs[1][1].draw_artist(vturb)
            fig.canvas.update()
            fig.canvas.flush_events()

            # plt.show()
            plt.savefig(write_path / 'plot_{}_{}_{}.png'.format(k, i + 662, j + 708), format='png', dpi=150)

            
 
plt.close('all')
plt.clf()
plt.cla()