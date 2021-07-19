from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


base_path = Path('/home/harsh/OsloAnalysis/new_kmeans/all_data_inversion_rps')

# falc_filename = base_path / 'RH15D_FALC_mu_0.7.nc'

quiet_filename = base_path / 'quiet/syn/wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63_syn.nc'

filenames = [
    base_path / 'wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63.nc',
    base_path / 'wholedata_rps_shock_profiles_rps_2_4_10_19_26_30_37_52_79_85_94_1_22_23_53_55_56_66_67_72_77_80_81_92_87_99_18_36_78_6_49_17_96_98.nc',
    base_path / 'wholedata_rps_reverse_shock_profiles_rps_3_13_16_25_32_33_35_41_45_46_58_61_64_68_82_95_97.nc',
    base_path / 'wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93.nc'
]

wave_ind = [
    np.array(
        [
            4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
            17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
            30,  31,  32,  36
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


wave_pind = [
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

wave_alt_pind = [
    np.array(range(0,29)),
    np.array(range(30, 50)),
    np.array(range(50, 64))
]

cont = [2.4434717e-08, 4.0148613e-08, 4.2277257e-08]


def do_correction():
    f = h5py.File(quiet_filename, 'r')
    fo = h5py.File(filenames[0], 'r')

    # a1 = f['alt_intensity'][29] / cont[0]
    # b1 = f['alt_intensity'][49] / cont[1]
    # c1 = f['alt_intensity'][63] / cont[2]
    # d1 = f['alt_intensity'][28] / cont[0]

    a1 = f['profiles'][0, 0, 2, wave_ind[0][-1], 0]
    b1 = f['profiles'][0, 0, 2, wave_ind[1][-1], 0]
    c1 = f['profiles'][0, 0, 2, wave_ind[2][-1], 0]
    d1 = f['profiles'][0, 0, 2, wave_ind[0][-2], 0]

    a2 = fo['profiles'][0, 0, 2, wave_ind[0][-1], 0]
    b2 = fo['profiles'][0, 0, 2, wave_ind[1][-1], 0]
    c2 = fo['profiles'][0, 0, 2, wave_ind[2][-1], 0]
    d2 = fo['profiles'][0, 0, 2, wave_ind[0][-2], 0]


    print (
        'a1: {}, b1: {}, c1: {}, d1: {}'.format(
            a1, b1, c1, d1
        )
    )

    print (
        'a2: {}, b2: {}, c2: {}, d2: {}'.format(
            a2, b2, c2, d2
        )
    )

#   a1: 0.6642471200855603, b1: 0.7168740850938748, c1: 0.8464003210886183, d1: 0.09549339422069777
#   a2: 0.6767058985179131, b2: 0.7762134127173193, c2: 1.0281220695362765, d2: 0.0956609904320442

    correction_factor = [
        b1 / b2,
        c1 / c2
    ]

    fo.close()
    f.close()

    f = h5py.File(quiet_filename, 'r')
    fo = h5py.File(filenames[0], 'r')

    fig, axs = plt.subplots(2, 2, figsize=(19.2, 10.8))

    # axs[0][0].plot(f['wav'][wave_alt_pind[0]], f['alt_intensity'][wave_alt_pind[0]] / cont[0], color='#3f3697')
    axs[0][0].plot(f['wav'][wave_pind[0]], f['profiles'][0, 0, 2, wave_pind[0], 0], color='#3f3697')
    axs[0][0].plot(fo['wav'][wave_pind[0]], fo['profiles'][0, 0, 2, wave_pind[0], 0], color='#c70039')


    # axs[0][1].scatter(f['wav'][29], f['alt_intensity'][29] / cont[0], color='#3f3697')
    axs[0][1].scatter(f['wav'][36], f['profiles'][0, 0, 2, 36, 0], color='#3f3697')
    axs[0][1].scatter(fo['wav'][36], fo['profiles'][0, 0, 2, 36, 0], color='#c70039')

    # axs[1][0].plot(f['wav'][wave_alt_pind[1]], f['alt_intensity'][wave_alt_pind[1]] / cont[1], color='#3f3697')
    axs[1][0].plot(f['wav'][wave_pind[1]], f['profiles'][0, 0, 2, wave_pind[1], 0], color='#3f3697')
    axs[1][0].plot(fo['wav'][wave_pind[1]], fo['profiles'][0, 0, 2, wave_pind[1], 0], color='#c70039')

    # axs[1][1].plot(f['wav'][wave_alt_pind[2]], f['alt_intensity'][wave_alt_pind[2]] / cont[2], color='#3f3697')
    axs[1][1].plot(f['wav'][wave_pind[2]], f['profiles'][0, 0, 2, wave_pind[2], 0], color='#3f3697')
    axs[1][1].plot(fo['wav'][wave_pind[2]], fo['profiles'][0, 0, 2, wave_pind[2], 0], color='#c70039')

    fig.savefig('Observation_vs_falc.png', format='png', dpi=300)

    f.close()
    fo.close()

    wei = np.ones((254, 4)) * 1e16
    wei[wave_ind[0], 0] = 0.001
    wei[wave_ind[0][9:19], 0] = 0.0005
    wei[wave_ind[1], 0] = 0.004
    wei[wave_ind[2], 0] = 0.002

    for filename in filenames:
        f = h5py.File(filename, 'r+')
        prof = f['profiles'][()]
        # prof[0, 0, :, wave_ind[0][0:-1], 0] *= correction_factor[2]
        prof[0, 0, :, wave_ind[1], 0] *= correction_factor[0]
        prof[0, 0, :, wave_ind[2], 0] *= correction_factor[1]
        f['profiles'][...] = prof
        f['weights'][...] = wei
        f['ori_wav'] = f['wav'][()]
        wave_new = f['wav'][()]
        # wave_new[146:] += 0.01
        # wave_new[0:36] -= 0.087
        print (wave_new[146])
        print (wave_new[0])
        f['wav'][...] = wave_new
        f.close()

    f = h5py.File(quiet_filename, 'r')
    fo = h5py.File(filenames[0], 'r')

    fig, axs = plt.subplots(2, 2, figsize=(19.2, 10.8))

    # axs[0][0].plot(f['wav'][wave_alt_pind[0]], f['alt_intensity'][wave_alt_pind[0]] / cont[0], color='#3f3697')
    axs[0][0].plot(f['wav'][wave_pind[0]], f['profiles'][0, 0, 2, wave_pind[0], 0], color='#3f3697')
    axs[0][0].plot(fo['wav'][wave_pind[0]], fo['profiles'][0, 0, 2, wave_pind[0], 0], color='#c70039')


    # axs[0][1].scatter(f['wav'][29], f['alt_intensity'][29] / cont[0], color='#3f3697')
    axs[0][1].scatter(f['wav'][36], f['profiles'][0, 0, 2, 36, 0], color='#3f3697')
    axs[0][1].scatter(fo['wav'][36], fo['profiles'][0, 0, 2, 36, 0], color='#c70039')

    # axs[1][0].plot(f['wav'][wave_alt_pind[1]], f['alt_intensity'][wave_alt_pind[1]] / cont[1], color='#3f3697')
    axs[1][0].plot(f['wav'][wave_pind[1]], f['profiles'][0, 0, 2, wave_pind[1], 0], color='#3f3697')
    axs[1][0].plot(fo['wav'][wave_pind[1]], fo['profiles'][0, 0, 2, wave_pind[1], 0], color='#c70039')

    # axs[1][1].plot(f['wav'][wave_alt_pind[2]], f['alt_intensity'][wave_alt_pind[2]] / cont[2], color='#3f3697')
    axs[1][1].plot(f['wav'][wave_pind[2]], f['profiles'][0, 0, 2, wave_pind[2], 0], color='#3f3697')
    axs[1][1].plot(fo['wav'][wave_pind[2]], fo['profiles'][0, 0, 2, wave_pind[2], 0], color='#c70039')

    fig.savefig('Corrected_Observation_vs_falc.png', format='png', dpi=300)

    f.close()
    fo.close()


if __name__ == '__main__':
    do_correction()
