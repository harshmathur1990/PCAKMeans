import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ltau = np.array(
    [
        -8.       , -7.78133  , -7.77448  , -7.76712  , -7.76004  ,
        -7.75249  , -7.74429  , -7.7356   , -7.72638  , -7.71591  ,
        -7.70478  , -7.69357  , -7.68765  , -7.68175  , -7.67589  ,
        -7.66997  , -7.66374  , -7.65712  , -7.64966  , -7.64093  ,
        -7.63093  , -7.6192   , -7.6053   , -7.58877  , -7.56925  ,
        -7.54674  , -7.52177  , -7.49317  , -7.4585   , -7.41659  ,
        -7.36725  , -7.31089  , -7.24834  , -7.18072  , -7.1113   ,
        -7.04138  , -6.97007  , -6.89698  , -6.82299  , -6.74881  ,
        -6.67471  , -6.60046  , -6.52598  , -6.45188  , -6.37933  ,
        -6.30927  , -6.24281  , -6.17928  , -6.11686  , -6.05597  ,
        -5.99747  , -5.94147  , -5.88801  , -5.84684  , -5.81285  ,
        -5.78014  , -5.74854  , -5.71774  , -5.68761  , -5.65825  ,
        -5.6293   , -5.60066  , -5.57245  , -5.54457  , -5.51687  ,
        -5.48932  , -5.46182  , -5.43417  , -5.40623  , -5.37801  ,
        -5.3496   , -5.32111  , -5.29248  , -5.26358  , -5.23413  ,
        -5.20392  , -5.17283  , -5.14073  , -5.1078   , -5.07426  ,
        -5.03999  , -5.00492  , -4.96953  , -4.93406  , -4.89821  ,
        -4.86196  , -4.82534  , -4.78825  , -4.75066  , -4.71243  ,
        -4.67439  , -4.63696  , -4.59945  , -4.5607   , -4.52212  ,
        -4.48434  , -4.44653  , -4.40796  , -4.36863  , -4.32842  ,
        -4.28651  , -4.24205  , -4.19486  , -4.14491  , -4.09187  ,
        -4.03446  , -3.97196  , -3.90451  , -3.83088  , -3.7496   ,
        -3.66     , -3.56112  , -3.4519   , -3.33173  , -3.20394  ,
        -3.07448  , -2.94444  , -2.8139   , -2.68294  , -2.55164  ,
        -2.42002  , -2.28814  , -2.15605  , -2.02377  , -1.89135  ,
        -1.7588   , -1.62613  , -1.49337  , -1.36127  , -1.23139  ,
        -1.10699  , -0.99209  , -0.884893 , -0.782787 , -0.683488 ,
        -0.584996 , -0.485559 , -0.383085 , -0.273456 , -0.152177 ,
        -0.0221309,  0.110786 ,  0.244405 ,  0.378378 ,  0.51182  ,
        0.64474  ,  0.777188 ,  0.909063 ,  1.04044  ,  1.1711
    ]
)


wave_list = [
    np.array(
        [
            3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
            3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
            3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
            3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
            3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
            3934.42352, 3934.48888, 3934.55424, 3934.6196
        ]
    ),
    np.array(
        [
            8540.3941552, 8540.9941552, 8541.2341552, 8541.3941552,
            8541.5541552, 8541.7141552, 8541.8341552, 8541.9141552,
            8541.9941552, 8542.0741552, 8542.1541552, 8542.2341552,
            8542.3141552, 8542.4341552, 8542.5941552, 8542.7541552,
            8542.9141552, 8543.1541552, 8543.7541552, 8544.4541552
        ]
    ),
    np.array(
        [
            6172.9802566, 6173.0602566, 6173.1402566, 6173.1802566,
            6173.2202566, 6173.2602566, 6173.3002566, 6173.3402566,
            6173.3802566, 6173.4202566, 6173.4602566, 6173.5402566,
            6173.6202566, 6173.9802566
        ]
    )
]


size = plt.rcParams['lines.markersize']


def get_all_profiles_and_atmos():
    subfoldername='other/plots_v1'
    base_path = Path('/home/harsh/OsloAnalysis/new_kmeans/all_data_inversion_rps')
    subfolder = base_path / subfoldername
    output_prof_name = subfolder / 'wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93_cycle_1_t_6_vl_5_vt_4_profs.nc'
    input_prof_name = base_path / 'wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93.nc'
    output_atmos_name = subfolder / 'wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93_cycle_1_t_6_vl_5_vt_4_atmos.nc'
    # output_atmos_name = base_path / 'wholedata_rps_initial_atmos_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63.nc'

    output_prof = h5py.File(output_prof_name, 'r')
    input_prof = h5py.File(input_prof_name, 'r')
    output_atmos = h5py.File(output_atmos_name, 'r')

    profiles = np.array(
        [
            5, 7, 9, 12, 27, 29, 38, 39, 50,
            54, 57, 59, 65, 71, 76, 83, 88, 91, 93
        ]
    )

    wave_ind = [
        np.array(
            [
                4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
                17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                30,  31,  32
            ]
        ),
        np.array(
            [
                41,  56,  62,  66,  70,  74,  77,  79,  81, 83,
                85,  87,  89,  92,  96, 100, 104, 110, 125, 143
            ]
        ),
        np.array(
            [
                150, 158, 166, 170, 174, 178, 182,
                186, 190, 194, 198, 206, 214, 250
            ]
        )
    ]

    cont_arr = (
        input_prof['profiles'][0, 0, :, 36, 0],
        output_prof['profiles'][0, 0, :, 36, 0]
    )

    row_1 = list()

    row_1.append(
        (
            input_prof['profiles'][0, 0, :, wave_ind[0], 0],
            output_prof['profiles'][0, 0, :, wave_ind[0], 0]
        )
    )

    row_1.append(
        output_atmos['temp'][0, 0, :]
    )

    row_2 = list()

    row_2.append(
        (
            input_prof['profiles'][0, 0, :, wave_ind[1], 0],
            output_prof['profiles'][0, 0, :, wave_ind[1], 0]
        )
    )

    row_2.append(
        output_atmos['vlos'][0, 0, :] / 1e5
    )

    row_3 = list()

    row_3.append(
        (
            input_prof['profiles'][0, 0, :, wave_ind[2], 0],
            output_prof['profiles'][0, 0, :, wave_ind[2], 0]
        )
    )

    row_3.append(
        output_atmos['vturb'][0, 0, :] / 1e5
    )

    row = [row_1, row_2, row_3]

    return profiles, row, cont_arr, 4001.14744, output_prof['wav'][36]


def make_plot():
    profiles, row, cont_arr, in_wave, out_wave = get_all_profiles_and_atmos()

    for k, profile in enumerate(list(profiles)):
        plt.close('all')
        plt.clf()
        plt.cla()
        fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69))
        for i in range(3):
            for j in range(2):
                if j == 0: 
                    axs[i][j].plot(wave_list[i], row[i][j][0][k], color='#c70039')
                    axs[i][j].scatter(wave_list[i], row[i][j][0][k], color='#c70039')
                    axs[i][j].plot(wave_list[i], row[i][j][1][k], color='#3f3697')
                    axs[i][j].scatter(wave_list[i], row[i][j][1][k], color='#3f3697')
                else:
                    axs[i][j].plot(ltau, row[i][j][k], color='#3f3697')
                if i != 2 or j != 1:
                    axs[i][j].set_xticklabels([])
        axs_2 = fig.add_subplot(321, label='alt', frame_on=False)
        axs_2.scatter(4001.14744, cont_arr[0][k], color='#c70039', marker='+')
        axs_2.scatter(4001.15, cont_arr[1][k], color='#3f3697', marker='+')
        axs_2.xaxis.tick_top()
        axs_2.xaxis.tick_top()
        axs_2.yaxis.tick_right()
        axs_2.yaxis.tick_right()
        axs_2.set_xticklabels([])

        print (
            'in_wave: {}'.format(
                in_wave
            )
        )
        print (
            'out_wave: {}'.format(
                out_wave
            )
        )

        fig.tight_layout()
        fig.savefig('RP_{}.eps'.format(profile), format='eps', dpi=300)
        fig.savefig('RP_{}.png'.format(profile), format='png', dpi=300)
