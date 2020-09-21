import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


base_input_path = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/no_vt_initial/custom_nodes/plots_v10/plots_rp_20_886'
)

# second_path = base_input_path / 'plots_v10_5'
profile_files = [
    base_input_path / 'rp_20_falc_vlos_guess_cycle_1_t_8_vl_8_vt_6_profs.nc',
]

atmos_files = [
    base_input_path / 'rp_20_falc_vlos_guess_cycle_1_t_8_vl_8_vt_6_atmos.nc',
]
observed_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/rps/observed_20.nc'
)
# observed_file = Path(
#     '/Users/harshmathur/CourseworkRepo/2008 Sp Data/stic_data_straylight_new_calculations/NICOLE_tries/buerro_approach/corrected_data_buererro_for_stic.nc'
# )
falc_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/model_atmos/falc_nicole_for_stic.nc'
)

median_profile_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/rps/observed_3.nc'
)

median_atmos_file = Path(
    '/Volumes/Harsh 9599771751/Oslo Work/no_vt_initial/custom_nodes/plots_v10/falc_vlos_guess_cycle_1_t_5_vl_2_vt_2_atmos.nc'
)

profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')
falc = h5py.File(falc_file, 'r')
median_profile = h5py.File(median_profile_file, 'r')
median_atmos = h5py.File(median_atmos_file, 'r')
indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]
# write_path = Path(
#     base_input_path / 'plots_v5'
# )
write_path = base_input_path # / 'plots_alternate'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

for i in range(1):

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 4)

    axs = [axs]

    # ------- Cycle 1 --------------------

    # plotting the observed profile
    axs[0][0].plot(
        observed['wav'][indices],
        observed['profiles'][0, 0, i, :, 0][indices],
        color=red,
        linewidth=0.5,
        label='Shock'
    )

    # plotting the inverted profile
    axs[0][0].plot(
        median_profile['wav'][:-1],
        profiles[0]['profiles'][0, 0, i, :-1, 0],
        color=green,
        linewidth=0.5,
        label='Fit'
    )

    axs[0][0].plot(
        median_profile['wav'][indices],
        median_profile['profiles'][0, 0, 0, :, 0][indices],
        color=brown,
        linewidth=0.5,
        label='QS Median'
    )

    axs[0][0].legend(loc='upper right')

    axs[0][0].set_ylim(0, 0.3)
    # plot FALC temperature profile
    axs[0][1].plot(
        median_atmos['ltau500'][0][0][3],
        median_atmos['temp'][0][0][3],
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
    axs[0][2].plot(
        median_atmos['ltau500'][0][0][3],
        median_atmos['vlos'][0][0][3] / 1e5,
        color=brown,
        linewidth=0.5
    )

    # plot inverted Vlos profile
    axs[0][2].plot(
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

    axs[0][2].set_ylim(-25, 25)

    # plot FALC Vturb profile
    axs[0][3].plot(
        median_atmos['ltau500'][0][0][3],
        median_atmos['vturb'][0][0][3] / 1e5,
        color=brown,
        linewidth=0.5
    )

    # plot inverted Vturb profile
    axs[0][3].plot(
        atmos[0]['ltau500'][0][0][i],
        atmos[0]['vturb'][0][0][i] / 1e5,
        color=green,
        linewidth=0.5
    )

    # axs[0][3].plot(
    #     atmos[0]['ltau500'][0][0][3],
    #     atmos[0]['vturb'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[0][3].set_ylim(0, 25)

    # ------- Cycle 1 --------------------

    # ------- Cycle 2 --------------------

    # plotting the observed profile
    # axs[1][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, i, :, 0][indices],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[1][0].plot(
    #     profiles[1]['wav'][:-1],
    #     profiles[1]['profiles'][0, 0, i, :-1, 0],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[1][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, 3, :, 0][indices],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[1][0].set_ylim(0, 0.3)
    # # plot FALC temperature profile
    # axs[1][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[1][1].plot(
    #     atmos[1]['ltau500'][0][0][i],
    #     atmos[1]['temp'][0][0][i],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[1][1].plot(
    #     atmos[1]['ltau500'][0][0][3],
    #     atmos[1]['temp'][0][0][3],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[1][1].set_ylim(4000, 11000)
    # # plot FALC Vlos profile
    # axs[1][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[1][2].plot(
    #     atmos[1]['ltau500'][0][0][i],
    #     atmos[1]['vlos'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[1][2].plot(
    #     atmos[1]['ltau500'][0][0][3],
    #     atmos[1]['vlos'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[1][2].set_ylim(-25, 25)

    # # plot FALC Vturb profile
    # axs[1][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[1][3].plot(
    #     atmos[1]['ltau500'][0][0][i],
    #     atmos[1]['vturb'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[1][3].plot(
    #     atmos[1]['ltau500'][0][0][3],
    #     atmos[1]['vturb'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[1][3].set_ylim(0, 25)

    # # ------- Cycle 2 --------------------

    # # ------- Cycle 3 --------------------

    # # plotting the observed profile
    # axs[2][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, i, :, 0][indices],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[2][0].plot(
    #     profiles[2]['wav'][:-1],
    #     profiles[2]['profiles'][0, 0, i, :-1, 0],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[2][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, 3, :, 0][indices],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[2][0].set_ylim(0, 0.3)

    # # plot FALC temperature profile
    # axs[2][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[2][1].plot(
    #     atmos[2]['ltau500'][0][0][i],
    #     atmos[2]['temp'][0][0][i],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[2][1].plot(
    #     atmos[2]['ltau500'][0][0][3],
    #     atmos[2]['temp'][0][0][3],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[2][1].set_ylim(4000, 11000)

    # # plot FALC Vlos profile
    # axs[2][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[2][2].plot(
    #     atmos[2]['ltau500'][0][0][i],
    #     atmos[2]['vlos'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[2][2].plot(
    #     atmos[2]['ltau500'][0][0][3],
    #     atmos[2]['vlos'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[2][2].set_ylim(-25, 25)

    # # plot FALC Vturb profile
    # axs[2][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[2][3].plot(
    #     atmos[2]['ltau500'][0][0][i],
    #     atmos[2]['vturb'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[2][3].plot(
    #     atmos[2]['ltau500'][0][0][3],
    #     atmos[2]['vturb'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[2][3].set_ylim(0, 25)

    # ------- Cycle 3 --------------------

    # # ------- Cycle 4 --------------------

    # # plotting the observed profile
    # axs[3][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, i, :, 0][indices],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[3][0].plot(
    #     profiles[3]['wav'][:-1],
    #     profiles[3]['profiles'][0, 0, i, :-1, 0],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[3][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, 3, :, 0][indices],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[3][0].set_ylim(0, 0.3)

    # # plot FALC temperature profile
    # axs[3][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[3][1].plot(
    #     atmos[3]['ltau500'][0][0][i],
    #     atmos[3]['temp'][0][0][i],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[3][1].plot(
    #     atmos[3]['ltau500'][0][0][3],
    #     atmos[3]['temp'][0][0][3],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[3][1].set_ylim(4000, 11000)

    # # plot FALC Vlos profile
    # axs[3][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[3][2].plot(
    #     atmos[3]['ltau500'][0][0][i],
    #     atmos[3]['vlos'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[3][2].plot(
    #     atmos[3]['ltau500'][0][0][3],
    #     atmos[3]['vlos'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[3][2].set_ylim(-25, 25)

    # # plot FALC Vturb profile
    # axs[3][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[3][3].plot(
    #     atmos[3]['ltau500'][0][0][i],
    #     atmos[3]['vturb'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[3][3].plot(
    #     atmos[3]['ltau500'][0][0][3],
    #     atmos[3]['vturb'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[3][3].set_ylim(0, 25)

    # ------- Cycle 4 --------------------

    # # ------- Cycle 5 --------------------

    # # plotting the observed profile
    # axs[4][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, i, :, 0][indices],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plotting the inverted profile
    # axs[4][0].plot(
    #     profiles[4]['wav'][:-1],
    #     profiles[4]['profiles'][0, 0, i, :-1, 0],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[4][0].plot(
    #     observed['wav'][indices],
    #     observed['profiles'][0, 0, 3, :, 0][indices],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[4][0].set_ylim(0, 0.3)

    # # plot FALC temperature profile
    # axs[4][1].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['temp'][0][0][0],
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted temperature profile
    # axs[4][1].plot(
    #     atmos[4]['ltau500'][0][0][i],
    #     atmos[4]['temp'][0][0][i],
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[4][1].plot(
    #     atmos[4]['ltau500'][0][0][3],
    #     atmos[4]['temp'][0][0][3],
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[4][1].set_ylim(4000, 11000)

    # # plot FALC Vlos profile
    # axs[4][2].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vlos'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vlos profile
    # axs[4][2].plot(
    #     atmos[4]['ltau500'][0][0][i],
    #     atmos[4]['vlos'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[4][2].plot(
    #     atmos[4]['ltau500'][0][0][3],
    #     atmos[4]['vlos'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[4][2].set_ylim(-25, 25)

    # # plot FALC Vturb profile
    # axs[4][3].plot(
    #     falc['ltau500'][0][0][0],
    #     falc['vturb'][0][0][0] / 1e5,
    #     color=red,
    #     linewidth=0.5
    # )

    # # plot inverted Vturb profile
    # axs[4][3].plot(
    #     atmos[4]['ltau500'][0][0][i],
    #     atmos[4]['vturb'][0][0][i] / 1e5,
    #     color=green,
    #     linewidth=0.5
    # )

    # axs[4][3].plot(
    #     atmos[4]['ltau500'][0][0][3],
    #     atmos[4]['vturb'][0][0][3] / 1e5,
    #     color=brown,
    #     linewidth=0.5
    # )

    # axs[4][3].set_ylim(0, 25)

    # # ------- Cycle 5 --------------------

    fig.tight_layout()

    # plt.show()
    plt.savefig(write_path / 'plot_{}.png'.format(i), format='png', dpi=1200)
