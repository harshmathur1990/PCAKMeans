import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

base_input_path = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/mean_profiles'
)

rp = base_input_path / 'all_rps'
second_path = rp / 'plots_v1'
profile_files = [
    second_path / 'rp_all_cycle_1_t_5_vl_1_vt_4_profs.nc',
]

atmos_files = [
    second_path / 'rp_all_falc_cycle_1_t_5_vl_1_vt_4_atmos.nc',
]
observed_file = Path(
    rp / 'merged_rps.nc'
)
# observed_file = Path(
#     '/Users/harshmathur/CourseworkRepo/2008 Sp Data/stic_data_straylight_new_calculations/NICOLE_tries/buerro_approach/corrected_data_buererro_for_stic.nc'
# )
falc_file = Path(
    base_input_path / 'falc_nicole_for_stic.nc'
)

# median_profile_file = Path(
#     base_input_path / 'observed_3.nc'
# )

# median_atmos_file = Path(
#     base_input_path / 'falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_atmos.nc'
# )

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
falc = h5py.File(falc_file, 'r')
# median_profile = h5py.File(median_profile_file, 'r')
# median_atmos = h5py.File(median_atmos_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]
# write_path = Path(
#     base_input_path / 'plots_v5'
# )
write_path = second_path # / 'plots_alternate'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

size = plt.rcParams['lines.markersize']

fontP = FontProperties()
fontP.set_size('xx-small')

# folder_list = [0, 4, 16, 23, 26, 33, 37, 41]
for i in range(45):

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(2, 2)

    # axs = [axs]

    # ------- Cycle 1 --------------------

    # plotting the observed profile
    axs[0][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red,
        linewidth=0.5,
        label='Shock'
    )

    axs[0][0].scatter(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red,
        s=size / 4
        # linewidth=0.5
        # label='Shock'
    )

    # plotting the inverted profile
    axs[0][0].plot(
        observed['wav'][:-1],
        profiles[0]['profiles'][0, 0, i, :-1, 0],
        color=green,
        linewidth=0.5,
        label='Fit'
    )

    axs[0][0].scatter(
        observed['wav'][:-1],
        profiles[0]['profiles'][0, 0, i, :-1, 0],
        color=green,
        # linewidth=0.5,
        s=size / 4
        # label='Fit'
    )

    axs[0][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, 3, :, 0][indices],
        color=brown,
        linewidth=0.5,
        label='QS Median'
    )

    axs[0][0].set_ylim(0, 0.3)
    # plot FALC temperature profile
    axs[0][1].plot(
        atmos[0]['ltau500'][0][0][3],
        atmos[0]['temp'][0][0][3],
        color=brown,
        linewidth=0.5
    )

    # plot inverted temperature profile
    axs[0][1].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['temp'][0][0][i],
        color=green,
        linewidth=0.5
    )

    # axs[0][1].plot(
    #     atmos[0]['ltau500'][0][0][3],
    #     atmos[0]['temp'][0][0][3],
    #     color=brown,
    #     linewidth=0.5
    # )

    axs[0][1].set_ylim(4000, 11000)

    # plot FALC Vlos profile
    axs[1][0].plot(
        atmos[0]['ltau500'][0][0][3],
        atmos[0]['vlos'][0][0][3] / 1e5,
        color=brown,
        linewidth=0.5
    )

    # plot inverted Vlos profile
    axs[1][0].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['vlos'][0][0][i] / 1e5,
        color=green,
        linewidth=0.5
    )

    # axs[0][2].plot(
    #     atmos[0]['ltau500'][0][0][3],
    #     atmos[0]['vlos'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    axs[1][0].set_ylim(0, 18)

    # plot FALC Vturb profile
    axs[1][1].plot(
        atmos[0]['ltau500'][0][0][3],
        atmos[0]['vturb'][0][0][3] / 1e5,
        color=brown,
        linewidth=0.5
    )

    # plot inverted Vturb profile
    axs[1][1].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['vturb'][0][0][i] / 1e5,
        color=green,
        linewidth=0.5
    )

    fig.tight_layout()

    axs[0][0].legend(loc='upper right', prop=fontP)
    # plt.show()
    plt.savefig(write_path / 'plot_{}.png'.format(i), format='png', dpi=1200)

    plt.close('all')
    plt.clf()
    plt.cla()
 