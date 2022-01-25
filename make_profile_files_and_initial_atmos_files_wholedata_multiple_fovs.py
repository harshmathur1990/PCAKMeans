import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
from prepare_data import *
from pathlib import Path
import sunpy.io
import h5py
from helita.io.lp import *

base_path = Path('/home/harsh/OsloAnalysis')

input_file_3950 = base_path / 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = base_path / 'nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = base_path / 'nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'

new_kmeans = base_path / 'new_kmeans'

all_data_inversion_rps = new_kmeans / 'all_data_inversion_rps'

label_file = base_path / 'new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

cw = np.asarray([4000., 6173., 8542.])
cont = []
for ii in cw:
    cont.append(getCont(ii))


b1 = 0.7168740850938748
c1 = 0.8464003210886183
b2 = 0.7762134127173193
c2 = 1.0281220695362765

correction_factor = [
    b1 / b2,
    c1 / c2
]

wave_3933 = np.array(
    [
        3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
        3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
        3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
        3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
        3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
        3934.42352, 3934.48888, 3934.55424, 3934.6196, 4001.14744
    ]
)

wave_8542 = np.array(
    [
        8540.3941552, 8540.9941552, 8541.2341552, 8541.3941552,
        8541.5541552, 8541.7141552, 8541.8341552, 8541.9141552,
        8541.9941552, 8542.0741552, 8542.1541552, 8542.2341552,
        8542.3141552, 8542.4341552, 8542.5941552, 8542.7541552,
        8542.9141552, 8543.1541552, 8543.7541552, 8544.4541552
    ]
)

wave_6173 = np.array(
    [
        6172.9802566, 6173.0602566, 6173.1402566, 6173.1802566,
        6173.2202566, 6173.2602566, 6173.3002566, 6173.3402566,
        6173.3802566, 6173.4202566, 6173.4602566, 6173.5402566,
        6173.6202566, 6173.9802566
    ]
)

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

pgas = np.array(
    [
        1.92480162e-01, 2.10335478e-01, 2.11099088e-01, 2.11936578e-01,
        2.12759241e-01, 2.13655323e-01, 2.14651138e-01, 2.15732709e-01,
        2.16910645e-01, 2.18287319e-01, 2.19797805e-01, 2.21369892e-01,
        2.22221285e-01, 2.23084763e-01, 2.23957345e-01, 2.24854335e-01,
        2.25815415e-01, 2.26856306e-01, 2.28054076e-01, 2.29489878e-01,
        2.31181175e-01, 2.33230412e-01, 2.35753953e-01, 2.38895416e-01,
        2.42812648e-01, 2.47627556e-01, 2.53370225e-01, 2.60512590e-01,
        2.70063460e-01, 2.83080816e-01, 3.00793827e-01, 3.24782073e-01,
        3.57132912e-01, 4.00601238e-01, 4.56965476e-01, 5.29293299e-01,
        6.24008834e-01, 7.49855459e-01, 9.16294158e-01, 1.13508153e+00,
        1.42164230e+00, 1.79735219e+00, 2.28865838e+00, 2.92158079e+00,
        3.71474624e+00, 4.67952347e+00, 5.81029558e+00, 7.12056494e+00,
        8.65885639e+00, 1.04314861e+01, 1.24189425e+01, 1.46121168e+01,
        1.69987698e+01, 1.90496445e+01, 2.08928623e+01, 2.28027058e+01,
        2.47815571e+01, 2.68437004e+01, 2.89950733e+01, 3.12255573e+01,
        3.35608482e+01, 3.60105362e+01, 3.85653992e+01, 4.12354431e+01,
        4.40376701e+01, 4.69793587e+01, 5.00764923e+01, 5.33600731e+01,
        5.68586159e+01, 6.05851173e+01, 6.45416412e+01, 6.87255859e+01,
        7.31585388e+01, 7.78767090e+01, 8.29482269e+01, 8.84405975e+01,
        9.44148178e+01, 1.00943398e+02, 1.08040993e+02, 1.15709175e+02,
        1.24027237e+02, 1.33072845e+02, 1.42778915e+02, 1.53122635e+02,
        1.64239197e+02, 1.76195251e+02, 1.89045700e+02, 2.02897873e+02,
        2.17846008e+02, 2.34042114e+02, 2.51210236e+02, 2.69186646e+02,
        2.88341248e+02, 3.09395111e+02, 3.31706787e+02, 3.54934540e+02,
        3.79624237e+02, 4.06384979e+02, 4.35409149e+02, 4.67003418e+02,
        5.02125732e+02, 5.41979980e+02, 5.87387085e+02, 6.39156738e+02,
        6.98575623e+02, 7.68398621e+02, 8.51376160e+02, 9.49656250e+02,
        1.06806201e+03, 1.21330579e+03, 1.39257166e+03, 1.61584387e+03,
        1.89664856e+03, 2.25192163e+03, 2.69018799e+03, 3.20748486e+03,
        3.81380298e+03, 4.52528320e+03, 5.36149463e+03, 6.34581787e+03,
        7.50641797e+03, 8.87612695e+03, 1.04934688e+04, 1.24038418e+04,
        1.46598135e+04, 1.73236582e+04, 2.04680117e+04, 2.41766445e+04,
        2.85175430e+04, 3.35073477e+04, 3.90268633e+04, 4.48068984e+04,
        5.07996992e+04, 5.70373594e+04, 6.35809258e+04, 7.05180547e+04,
        7.79622109e+04, 8.61010938e+04, 9.53656875e+04, 1.06372070e+05,
        1.19236258e+05, 1.33759531e+05, 1.49984500e+05, 1.68110281e+05,
        1.88273656e+05, 2.10752016e+05, 2.35874125e+05, 2.63987969e+05,
        2.95532281e+05, 3.30931750e+05
    ]
)

new_quiet_profiles = np.array([0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63])
new_shock_reverse_other_profiles = np.array([2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98, 3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97, 5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93])
shocks_78_18 = np.array([78, 18])

names = ['quiet_profiles_rps', 'shock_reverse_other_profiles_rps', 'shocks_78_18_profiles_rps']
write_path = None
x = None
y = None


def get_data():
    data_3950, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    sh, dt, header = getheader(input_file_6173)
    data_6173 = np.memmap(
        input_file_6173,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data_6173 = np.transpose(
        data_6173.reshape(1848, 1236, 100, 4, 14),
        axes=(2, 3, 4, 1, 0)
    )

    sh, dt, header = getheader(input_file_8542)
    data_8542 = np.memmap(
        input_file_8542,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data_8542 = np.transpose(
        data_8542.reshape(1848, 1236, 100, 4, 20),
        axes=(2, 3, 4, 1, 0)
    )

    return data_3950, data_8542, data_6173


def get_atmos_values_for_lables():

    temp = np.zeros((100, 150))

    vlos = np.zeros((100, 150))

    vturb = np.zeros((100, 150))

    for i in range(100):
        filename, content_list = get_filepath_and_content_list(i)

        f = h5py.File(filename, 'r')

        index = content_list.index(i)

        if len(content_list) == f['ltau500'].shape[1]:
            normal = True
        else:
            normal = False

        if normal:

            temp[i] = f['temp'][0, index, 0]

            vlos[i] = f['vlos'][0, index, 0]

            vturb[i] = f['vturb'][0, index, 0]

        else:

            temp[i] = f['temp'][0, 0, index]

            vlos[i] = f['vlos'][0, 0, index]

            vturb[i] = f['vturb'][0, 0, index]

        f.close()

    return temp, vlos, vturb


def prepare_get_parameter(param):

    def get_parameter(rp):
        return param[rp]

    return get_parameter


def get_filepath_and_content_list(rp):
    if rp in [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]:
        filename, content_list = all_data_inversion_rps / 'quiet/plots_v1/wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63_cycle_1_t_6_vl_3_vt_4_atmos.nc', [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]

    elif rp in [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]:
        filename, content_list = all_data_inversion_rps / 'shocks/plots_v1/wholedata_rps_shock_profiles_rps_2_4_10_19_26_30_37_52_79_85_94_1_22_23_53_55_56_66_67_72_77_80_81_92_87_99_36_6_49_17_96_98_cycle_1_t_6_vl_5_vt_4_atmos.nc', [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]

    elif rp in [78, 18]:
        filename, content_list = all_data_inversion_rps / 'shocks_78_18/plots_v1/wholedata_rps_shocks_78_18_profile_rps_78_18_cycle_1_t_5_vl_5_vt_4_atmos.nc', [78, 18]

    elif rp in [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]:
        filename, content_list = all_data_inversion_rps / 'reverse/plots_v1/wholedata_rps_reverse_shock_profiles_rps_3_13_16_25_32_33_35_41_45_46_58_61_64_68_82_95_97_cycle_1_t_6_vl_5_vt_4_atmos.nc', [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]

    elif rp in [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]:
        filename, content_list = all_data_inversion_rps / 'other/plots_v1/wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93_cycle_1_t_6_vl_5_vt_4_atmos.nc', [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]

    return filename, content_list


def get_name_string(fov_list):

    # str_list = list()
    #
    # for fov in fov_list:
    #     x = fov[0]
    #
    #     y = fov[1]
    #
    #     t = fov[2]
    #
    #     strs = 'x_{}_{}_y_{}_{}_frame_{}_{}'.format(
    #         x[0], x[1], y[0], y[1], t[0], t[1]
    #     )
    #
    #     str_list.append(strs)
    #
    # return '_'.join(str_list)

    return 'more_frames_for_fov'


def get_data_for_inversions(fov_list, total_frames=23):
    input_profiles = np.zeros(
        (total_frames, 50, 50, 64),
        dtype=np.float64
    )

    input_labels = np.zeros(
        (total_frames, 50, 50),
        dtype=np.int64
    )

    data_3950, data_8542, data_6173 = get_data()

    f = h5py.File(label_file, 'r')

    k = 0

    for index, fov in enumerate(fov_list):
        x = fov[0]

        y = fov[1]

        t = fov[2]

        input_profiles[k: k + t[1]-t[0], :, :, 0:30] = np.transpose(
            data_3950[t[0]:t[1], 0, :, x[0]:x[1], y[0]:y[1]],
            axes=(0, 2, 3, 1)
        )
        input_profiles[k: k + t[1]-t[0], :, :, 30:30 + 14] = np.transpose(
            data_6173[t[0]:t[1], 0, :, x[0]:x[1], y[0]:y[1]],
            axes=(0, 2, 3, 1)
        )
        input_profiles[k: k + t[1]-t[0], :, :, 30 + 14:30 + 14 + 20] = np.transpose(
            data_8542[t[0]:t[1], 0, :, x[0]:x[1], y[0]:y[1]],
            axes=(0, 2, 3, 1)
        )

        input_labels[k: k + t[1]-t[0]] = f['new_final_labels'][t[0]:t[1], x[0]:x[1], y[0]:y[1]]

        k += t[1] - t[0]

    f.close()

    return input_profiles, input_labels


def make_files(fov_list, frs, total_frames=23):

    global write_path

    input_profiles, input_labels = get_data_for_inversions(fov_list, total_frames)

    name_string = get_name_string(fov_list)

    wck, ick = findgrid(wave_3933[:-1], (wave_3933[1] - wave_3933[0]), extra=8)
    wfe, ife = findgrid(wave_6173, (wave_6173[10] - wave_6173[9])*0.25, extra=8)
    wc8, ic8 = findgrid(wave_8542, (wave_8542[10] - wave_8542[9])*0.5, extra=8)

    for profile_type, name, fr in zip([new_quiet_profiles, new_shock_reverse_other_profiles, shocks_78_18], names, frs):

        for frame in fr:
            a_final, b_final, c_final, rp_final = list(), list(), list(), list()

            for profile in profile_type:
                a, b, c = np.where(input_labels[frame[0]:frame[1]] == profile)
                a_final += list(a)
                b_final += list(b)
                c_final += list(c)
                rp_final += list(np.ones(a.shape[0]) * profile)

            a_final = np.array(a_final)

            b_final = np.array(b_final)

            c_final = np.array(c_final)

            rp_final = np.array(rp_final)

            if a_final.size == 0:
                continue
            pixel_indices = np.zeros((4, a_final.size), dtype=np.int64)

            pixel_indices[0] = a_final
            pixel_indices[1] = b_final
            pixel_indices[2] = c_final
            pixel_indices[3] = rp_final

            fo = h5py.File(
                write_path / 'pixel_indices_{}_frame_{}_{}_{}_total_{}.h5'.format(
                    name, frame[0], frame[1], name_string, a_final.size
                ), 'w'
            )

            fo['pixel_indices'] = pixel_indices

            fo.close()

            ca_k = sp.profile(nx=a_final.size, ny=1, ns=4, nt=1, nw=wck.size+1)
            fe_1 = sp.profile(nx=a_final.size, ny=1, ns=4, nw=wfe.size)
            ca_8 = sp.profile(nx=a_final.size, ny=1, ns=4, nw=wc8.size)

            fe_1.wav[:] = wfe[:]
            ca_8.wav[:] = wc8[:]
            ca_k.wav[0:-1] = wck[:]
            ca_k.wav[-1]    = wave_3933[-1]

            ca_k.dat[0, 0, :, ick, 0] = input_profiles[frame[0]:frame[1]][a_final, b_final, c_final, 0:29].T / cont[0]

            ca_k.dat[0, 0, :, -1, 0] = input_profiles[frame[0]:frame[1]][a_final, b_final, c_final, 29].T / cont[0]

            fe_1.dat[0, 0, :, ife, 0] = input_profiles[frame[0]:frame[1]][a_final, b_final, c_final, 30:30 + 14].T * 1e3 / cont[1]

            fe_1.dat[0, 0, :, ife, 0] *= correction_factor[1]

            ca_8.dat[0,0,:,ic8, 0] = input_profiles[frame[0]:frame[1]][a_final, b_final, c_final, 30 + 14:30 + 14 + 20].T * 1e3 / cont[2]

            ca_8.dat[0, 0, :, ic8, 0] *= correction_factor[0]

            fe_1.weights[:,:] = 1.e16
            fe_1.weights[ife, 0] = 0.002

            ca_8.weights[:,:] = 1.e16
            ca_8.weights[ic8, 0] = 0.004
            
            ca_k.weights[:,:] = 1.e16
            ca_k.weights[ick,0] = 0.001
            ca_k.weights[ick[9:19],0] = 0.0005
            ca_k.weights[-1,0] = 0.001

            write_filename = write_path / 'wholedata_{}_frame_{}_{}_{}_total_{}.nc'.format(
                name, frame[0], frame[1], name_string, a_final.size
            )

            sp_all = ca_k + ca_8 + fe_1

            sp_all.write(str(write_filename))

            labels = rp_final.astype(np.int64)

            m = sp.model(nx=a_final.size, ny=1, nt=1, ndep=150)

            temp, vlos, vturb = get_atmos_values_for_lables()

            get_temp = prepare_get_parameter(temp)

            get_vlos = prepare_get_parameter(vlos)

            get_vturb = prepare_get_parameter(vturb)

            m.ltau[:, :, :] = ltau

            m.pgas[:, :, :] = 0.3

            m.temp[0, 0] = get_temp(labels)

            m.vlos[0, 0] = get_vlos(labels)

            m.vturb[0, 0] = get_vturb(labels)

            write_filename = write_path / 'wholedata_{}_initial_atmos_frame_{}_{}_{}_total_{}.nc'.format(
                name, frame[0], frame[1], name_string, a_final.size
            )

            m.write(str(write_filename))


if __name__ == '__main__':

    # fov_list = [
    #     ([915, 965], [1072, 1122], [14, 21]),
    #     ([486, 536], [974, 1024], [17, 24]),
    #     ([582, 632], [627, 677], [32, 39]),
    #     ([810, 860], [335, 385], [12, 19]),
    #     ([455, 505], [940, 990], [57, 64]),
    #     ([95, 145], [600, 650], [93, 100]),
    #     ([315, 365], [855, 905], [7, 14]),
    #     ([600, 650], [1280, 1330], [8, 15])
    # ]

    # fov_list = [
    #     ([535, 585], [715, 765], [9, 16]),
    # ]

    # fov_list = [
    #     ([810, 860], [335, 385], [10, 12])
    # ]

    fov_list = [
        ([662, 712], [708, 758], [3, 4]),  # A
        ([662, 712], [708, 758], [11, 13]),  # A
        ([582, 632], [627, 677], [30, 32]),  # C
        ([582, 632], [627, 677], [39, 41]),  # C
        ([810, 860], [335, 385], [11, 12]),  # D
        ([810, 860], [335, 385], [19, 22]),  # D
        ([315, 365], [855, 905], [6, 7]),  # F
        ([315, 365], [855, 905], [14, 17]),  # F
        ([600, 650], [1280, 1330], [6, 8]),  # G
        ([600, 650], [1280, 1330], [15, 18]),  # G
        ([535, 585], [715, 765], [8, 9]),  # H
        ([535, 585], [715, 765], [16, 18]),  # H
    ]
    write_path = base_path / 'new_kmeans/wholedata_inversions/fov_more/'

    quiet_frames_list = [[0, 23]]
    shock_reverse_other_frames_list = [[0, 23]]
    shocks_78_18_frames_list = [[0, 23]]
    make_files(
        fov_list,
        frs=(
            quiet_frames_list,
            shock_reverse_other_frames_list,
            shocks_78_18_frames_list
        ),
        total_frames=23
    )
