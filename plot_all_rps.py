import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


base_input_path = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/all_representative_profile_inversions'
)
profile_files = [
    base_input_path / 'cycle_1_profs_sw_t_7_vt_0_vl_3.nc',
    base_input_path / 'cycle_2_profs_w_16_23_t_7_vt_0_vl_3.nc',
    base_input_path / 'cycle_3_profs_sw_t_7_vt_5_vl_3.nc'
]
atmos_files = [
    base_input_path / 'cycle_1_atmos_sw_t_7_vt_0_vl_3.nc',
    base_input_path / 'cycle_2_atmos_w_16_23_t_7_vt_0_vl_3.nc',
    base_input_path / 'cycle_3_atmos_sw_t_7_vt_5_vl_3.nc'
]
observed_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/merged_rps.nc'
)
falc_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/model_atmos/falc_nicole_for_stic.nc'
)

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
falc = h5py.File(falc_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]
write_path = Path(
    base_input_path / 'plots'
)

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

for i in range(45):

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(3, 4)

    # ------- Cycle 1 --------------------

    # plotting the observed profile
    axs[0][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red
    )

    # plotting the inverted profile
    axs[0][0].plot(
        profiles[0]['wav'][:-1],
        profiles[0]['profiles'][0, 0, i, :-1, 0],
        color=green
    )

    axs[0][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, 3, :, 0][indices],
        color=brown
    )

    axs[0][0].set_ylim(0, 0.3)
    # plot FALC temperature profile
    axs[0][1].plot(
        falc['ltau500'][0][0][0],
        falc['temp'][0][0][0],
        color=red
    )

    # plot inverted temperature profile
    axs[0][1].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['temp'][0][0][i],
        color=green
    )

    axs[0][1].plot(
        atmos[0]['ltau500'][0][0][3],
        atmos[0]['temp'][0][0][3],
        color=brown
    )

    axs[0][1].set_ylim(0, 11000)

    # plot FALC Vlos profile
    axs[0][2].plot(
        falc['ltau500'][0][0][0],
        falc['vlos'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vlos profile
    axs[0][2].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['vlos'][0][0][i] / 1e4,
        color=green
    )

    axs[0][2].set_ylim(-20, 100)

    # plot FALC Vturb profile
    axs[0][3].plot(
        falc['ltau500'][0][0][0],
        falc['vturb'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vturb profile
    axs[0][3].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['vturb'][0][0][i] / 1e4,
        color=green
    )

    axs[0][3].set_ylim(0, 80)

    # ------- Cycle 1 --------------------

    # ------- Cycle 2 --------------------

    # plotting the observed profile
    axs[1][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red
    )

    # plotting the inverted profile
    axs[1][0].plot(
        profiles[1]['wav'][:-1],
        profiles[1]['profiles'][0, 0, i, :-1, 0],
        color=green
    )

    axs[1][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, 3, :, 0][indices],
        color=brown
    )

    axs[1][0].set_ylim(0, 0.3)
    # plot FALC temperature profile
    axs[1][1].plot(
        falc['ltau500'][0][0][0],
        falc['temp'][0][0][0],
        color=red
    )

    # plot inverted temperature profile
    axs[1][1].plot(
        atmos[1]['ltau500'][0][0][i],
        atmos[1]['temp'][0][0][i],
        color=green
    )

    axs[1][1].plot(
        atmos[1]['ltau500'][0][0][3],
        atmos[1]['temp'][0][0][3],
        color=brown
    )

    axs[1][1].set_ylim(0, 11000)
    # plot FALC Vlos profile
    axs[1][2].plot(
        falc['ltau500'][0][0][0],
        falc['vlos'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vlos profile
    axs[1][2].plot(
        atmos[1]['ltau500'][0][0][i],
        atmos[1]['vlos'][0][0][i] / 1e4,
        color=green
    )

    axs[1][2].set_ylim(-20, 100)

    # plot FALC Vturb profile
    axs[1][3].plot(
        falc['ltau500'][0][0][0],
        falc['vturb'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vturb profile
    axs[1][3].plot(
        atmos[1]['ltau500'][0][0][i],
        atmos[1]['vturb'][0][0][i] / 1e4,
        color=green
    )

    axs[1][3].set_ylim(0, 80)

    # ------- Cycle 2 --------------------

    # ------- Cycle 3 --------------------

    # plotting the observed profile
    axs[2][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red
    )

    # plotting the inverted profile
    axs[2][0].plot(
        profiles[2]['wav'][:-1],
        profiles[2]['profiles'][0, 0, i, :-1, 0],
        color=green
    )

    axs[2][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, 3, :, 0][indices],
        color=brown
    )

    axs[2][0].set_ylim(0, 0.3)

    # plot FALC temperature profile
    axs[2][1].plot(
        falc['ltau500'][0][0][0],
        falc['temp'][0][0][0],
        color=red
    )

    # plot inverted temperature profile
    axs[2][1].plot(
        atmos[2]['ltau500'][0][0][i],
        atmos[2]['temp'][0][0][i],
        color=green
    )

    axs[2][1].plot(
        atmos[2]['ltau500'][0][0][3],
        atmos[2]['temp'][0][0][3],
        color=brown
    )

    axs[2][1].set_ylim(0, 11000)

    # plot FALC Vlos profile
    axs[2][2].plot(
        falc['ltau500'][0][0][0],
        falc['vlos'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vlos profile
    axs[2][2].plot(
        atmos[2]['ltau500'][0][0][i],
        atmos[2]['vlos'][0][0][i] / 1e4,
        color=green
    )

    axs[2][2].set_ylim(-20, 100)

    # plot FALC Vturb profile
    axs[2][3].plot(
        falc['ltau500'][0][0][0],
        falc['vturb'][0][0][0] / 1e4,
        color=red
    )

    # plot inverted Vturb profile
    axs[2][3].plot(
        atmos[2]['ltau500'][0][0][i],
        atmos[2]['vturb'][0][0][i] / 1e4,
        color=green
    )

    axs[2][3].set_ylim(0, 80)

    # ------- Cycle 3 --------------------

    fig.tight_layout()

    plt.savefig(write_path / 'plot_{}.png'.format(i), format='png', dpi=300)
