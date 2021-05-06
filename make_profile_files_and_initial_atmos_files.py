import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
from prepare_data import *
from pathlib import Path
import sunpy.io
import h5py


base_path = Path('/home/harsh/OsloAnalysis')

new_kmeans = base_path / 'new_kmeans'

write_path = base_path / 'new_kmeans/inversions/plots_v1_fifth_fov'

data, header = sunpy.io.fits.read(base_path / 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits', memmap=True)[0]

label_file = base_path / 'new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

cw = np.asarray([4000.])
cont = []
for ii in cw:
    cont.append(getCont(ii))

## fov1
# frames = [0, 21]
# x = [662, 712]
# y = [708, 758]

## fov2
frames = [77, 100]
x = [770, 820]
y = [338, 388]

## fov3
# frames = [0, 21]
# x = [520, 570]
# y = [715, 765]

## fov4
# frames = [21, 100]
# x = [662, 712]
# y = [708, 758]

## fov5
# frames = [0, 21]
# x = [770, 820]
# y = [338, 388]

## fov6
# frames = [77, 100]
# x = [770, 820]
# y = [338, 388]

## fov7
# frames = [21, 100]
# x = [520, 570]
# y = [715, 765]


cw = np.asarray([4000.])
cont = []

for ii in cw: cont.append(getCont(ii))

wave = np.array(
    [
        3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
        3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
        3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
        3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
        3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
        3934.42352, 3934.48888, 3934.55424, 3934.6196 , 4001.14744
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

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 8, 44, 63, 84]

shock_spicule_profiles = [4, 10, 19, 26, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 72, 77, 92, 99]

retry_shock_spicule = [6, 49, 18, 36]

shock_78 = [78]

reverse_shock_profiles = [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

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
    if rp in quiet_profiles:
        filename, content_list = new_kmeans / 'quiet_profiles/plots_v1/rp_0_3_8_11_14_15_16_20_21_24_28_31_34_40_42_43_44_47_48_51_57_58_60_61_62_63_66_69_70_71_73_74_75_76_82_84_86_89_90_99_cycle_1_t_5_vl_1_vt_4_atmos.nc', [0, 3, 8, 11, 14, 15, 16, 20, 21, 24, 28, 31, 34, 40, 42, 43, 44, 47, 48, 51, 57, 58, 60, 61, 62, 63, 66, 69, 70, 71, 73, 74, 75, 76, 82, 84, 86, 89, 90, 99]

    elif rp in shock_spicule_profiles:
        filename, content_list = new_kmeans / 'shock_and_spicule_profiles/plots_v1/rp_1_4_6_10_15_18_19_20_22_23_24_26_36_37_40_43_49_52_53_55_56_62_66_70_72_73_74_75_77_78_79_84_85_86_92_94_99_cycle_1_t_4_vl_5_vt_4_atmos.nc', [1, 4, 6, 10, 15, 18, 19, 20, 22, 23, 24, 26, 36, 37, 40, 43, 49, 52, 53, 55, 56, 62, 66, 70, 72, 73, 74, 75, 77, 78, 79, 84, 85, 86, 92, 94, 99]

    elif rp in retry_shock_spicule:
        filename, content_list = new_kmeans / 'retry_shock_spicule/plots_v1/rp_6_49_18_36_78_cycle_1_t_4_vl_5_vt_4_atmos.nc', [6, 49, 18, 36, 78]

    elif rp in shock_78:
        filename, content_list = new_kmeans / 'rp_78/plots_v3/rp_78_cycle_1_t_4_vl_7_vt_4_atmos.nc', [78]

    elif rp in reverse_shock_profiles:
        filename, content_list = new_kmeans / 'reverse_shock_profiles/plots_v1/rp_3_13_16_17_25_32_33_35_41_58_61_64_68_82_95_97_cycle_1_t_5_vl_5_vt_4_atmos.nc', [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

    elif rp in other_emission_profiles:
        filename, content_list = new_kmeans / 'other_emission_profiles/plots_v1/rp_2_5_7_9_12_27_29_30_38_39_45_46_50_54_57_59_65_67_71_76_80_81_83_87_88_91_93_96_98_cycle_1_t_5_vl_5_vt_4_atmos.nc', [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

    return filename, content_list

wck, ick = findgrid(wave[:-1], (wave[1] - wave[0]), extra=8)

f = h5py.File(label_file, 'r')

names = ['quiet_profiles', 'shock_spicule_profiles', 'retry_shock_spicule_profiles', 'reverse_shock_profiles', 'shock_78_profiles', 'other_emission_profiles']

for profile_type, name in zip([quiet_profiles, shock_spicule_profiles, retry_shock_spicule, reverse_shock_profiles, shock_78, other_emission_profiles], names):

    a_final, b_final, c_final, rp_final = list(), list(), list(), list()

    for profile in profile_type:
        a, b, c = np.where(f['new_final_labels'][frames[0]:frames[1], x[0]:x[1], y[0]:y[1]] == profile)
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

    fo = h5py.File('pixel_indices_{}_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(name, frames[0], frames[1], x[0], x[1], y[0], y[1]), 'w')

    fo['pixel_indices'] = pixel_indices

    fo.close()

    ca_k = sp.profile(nx=a_final.size, ny=1, ns=4, nt=1, nw=wck.size+1)

    ca_k.wav[0:-1] = wck[:]

    ca_k.wav[-1] = wave[-1]

    ca_k.dat[0, 0, :, ick, 0] = data[frames[0]:frames[1], 0, :-1, x[0]:x[1], y[0]:y[1]][a_final, :, b_final, c_final].T / cont[0]

    ca_k.dat[0, 0, :, -1, 0] = data[frames[0]:frames[1], 0, :, x[0]:x[1], y[0]:y[1]][a_final, -1, b_final, c_final].T / cont[0]

    ca_k.weights[:,:] = 1.e16

    ca_k.weights[ick,0] = 0.002

    ca_k.weights[-1,0] = 0.004

    write_filename = write_path / '{}_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(name, frames[0], frames[1], x[0], x[1], y[0], y[1])

    ca_k.write(str(write_filename))

    labels = rp_final.astype(np.int64)

    m = sp.model(nx=a_final.size, ny=1, nt=1, ndep=150)

    temp, vlos, vturb = get_atmos_values_for_lables()

    get_temp = prepare_get_parameter(temp)

    get_vlos = prepare_get_parameter(vlos)

    get_vturb = prepare_get_parameter(vturb)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = pgas

    m.temp[0, 0] = get_temp(labels)

    m.vlos[0, 0] = get_vlos(labels)

    m.vturb[0, 0] = get_vturb(labels)

    write_filename = write_path / '{}_initial_atmos_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(name, frames[0], frames[1], x[0], x[1], y[0], y[1])

    m.write(str(write_filename))

