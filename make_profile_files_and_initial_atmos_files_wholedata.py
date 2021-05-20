import sys
sys.path.insert(1, '/home/harsh/stic/example')
from prepare_data import *
from pathlib import Path
import numpy as np
import h5py
import sunpy.io
from helita.io.lp import *


selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
old_kmeans_file = '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
mask_file_crisp = '/data/harsh/crisp_chromis_mask_2019-06-06.fits'
input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'

mask, _  = sunpy.io.fits.read(mask_file_crisp, memmap=True)[0]

mask = np.transpose(mask, axes=(2, 1, 0))

new_quiet_profiles = np.array([0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63])
new_shock_profiles = np.array([2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 18, 36, 78, 6, 49, 17, 96, 98])
new_reverse_shock_profiles = np.array([3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97])
new_other_emission_profiles = np.array([5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93])

names = ['quiet_profiles_rps', 'shock_profiles_rps', 'reverse_shock_profiles_rps', 'other_emission_profiles_rps']
wave_3933 = np.array(
    [
        3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
        3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
        3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
        3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
        3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
        3934.42352, 3934.48888, 3934.55424, 3934.6196
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

cont_value = [2.4434714e-05, 4.2277254e-08, 4.054384e-08]
cont_array = np.zeros(30 + 20 + 14)
cont_array[0:30] = cont_value[0]
cont_array[30:30 + 20] = cont_value[1]
cont_array[30 + 20: 30 + 20 + 14] = cont_value[2]
in_bins_3950 = np.linspace(0, 0.6, 1000)
in_bins_8542 = np.linspace(0, 1.3, 1000)
in_bins_6173 = np.linspace(0.4, 2, 1000)


cw = np.asarray([4000., 6302., 8542.])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def get_farthest(a, center):
    global whole_data
    all_profiles = whole_data[a] / cont_array
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index]


def get_old_rps_result_atmos():
    base_path = Path('/home/harsh/OsloAnalysis')

    new_kmeans = base_path / 'new_kmeans'

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

    temp, vlos, vturb = get_atmos_values_for_lables()

    get_temp = prepare_get_parameter(temp)

    get_vlos = prepare_get_parameter(vlos)

    get_vturb = prepare_get_parameter(vturb)

    return get_temp, get_vlos, get_vturb


def get_data():
    n, o, p = np.where(mask[selected_frames] == 1)

    whole_data = np.zeros((n.size, 30 + 14 + 20, 4))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[:, 0:30, 0] = data[selected_frames][n, 0, :, o, p]

    sh, dt, header = getheader(input_file_6173)
    data = np.memmap(
        input_file_6173,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 14),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[:, 30:30 + 14, :] = np.transpose(data[selected_frames][n, :, :, o, p], axes=(0, 2, 1))

    sh, dt, header = getheader(input_file_8542)
    data = np.memmap(
        input_file_8542,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 20),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[:, 30 + 14:30 + 14 + 20, :] = np.transpose(data[selected_frames][n, 0, :, o, p], axes=(0, 2, 1))

    return whole_data, n, o, p


def make_files():
    whole_data, n, o, p = get_data()

    f = h5py.File(old_kmeans_file, 'r')
    labels = f['new_final_labels'][selected_frames][n, o, p]
    rps = np.zeros((100, 64, 4))
    for i in range(100):
        ind = np.where(labels == i)[0]
        rps[i] = np.mean(whole_data[ind], 0)

    wck, ick = findgrid(wave_3933[:-1], (wave_3933[1] - wave_3933[0]), extra=8)
    wfe, ife = findgrid(wave_6173, (wave_6173[1] - wave_6173[0])*0.25, extra=8)
    wc8, ic8 = findgrid(wave_8542, (wave_8542[1] - wave_8542[0])*0.5, extra=8)

    get_temp, get_vlos, get_vturb = get_old_rps_result_atmos()

    for profiles, name in zip([new_quiet_profiles, new_shock_profiles, new_reverse_shock_profiles, new_other_emission_profiles], names):
        fe_1 = sp.profile(nx=profiles.size, ny=1, ns=4, nw=wfe.size)
        ca_8 = sp.profile(nx=profiles.size, ny=1, ns=4, nw=wc8.size)
        ca_k = sp.profile(nx=profiles.size, ny=1, ns=4, nw=wck.size+1)

        fe_1.wav[:] = wfe[:]
        ca_8.wav[:] = wc8[:]
        ca_k.wav[0:-1] = wck[:]
        ca_k.wav[-1]    = wave_3933[-1]

        fe_1.dat[0,0,:,ife,:] = np.transpose(
            rps[profiles, 30:30 + 14, :] * 1e3 / cont[1],
            axes=(1, 0, 2)
        )

        ca_8.dat[0,0,:,ic8,:] = np.transpose(
            rps[profiles, 30 + 14:30 + 14 + 20, :] * 1e3 / cont[2],
            axes=(1, 0, 2)
        )

        ca_k.dat[0,0,:,ick,:] = np.transpose(
            rps[profiles, 0:29, :] / cont[0],
            axes=(1, 0, 2)
        )

        ca_k.dat[0,0,:, -1,:] = rps[profiles, 29, :] / cont[0]

        fe_1.weights[:,:] = 1.e16 # Very high value means weight zero
        fe_1.weights[ife,:] = 0.005
        fe_1.weights[ife,1:3] /= 4.5 # Some more weight for Q&U
        fe_1.weights[ife,3] /= 3.5    # Some more weight for V

        ca_8.weights[:,:] = 1.e16 # Very high value means weight zero
        ca_8.weights[ic8,:] = 0.004
        ca_8.weights[ic8,1:3] /= 7.0 # Some more weight for Q&U
        ca_8.weights[ic8,3] /= 4.0    # Some more weight for V
        ca_8.weights[ic8[9:12],0] /= 2.0
        
        ca_k.weights[:,:] = 1.e16 # Very high value means weight zero
        ca_k.weights[ick,0] = 0.002
        ca_8.weights[ick[19:22],0] /= 2.0
        ca_k.weights[-1,0] = 0.004 # Continuum point

        sp_all = ca_k + ca_8 + fe_1
        sp_all.write(
            'wholedata_rps_{}_{}.nc'.format(
                name,
                '_'.join(
                    [
                        str(profile) for profile in list(profiles)
                    ]
                )
            )
        )

        m = sp.model(nx=profiles.size, ny=1, nt=1, ndep=150)

        m.ltau[:, :, :] = ltau

        m.pgas[:, :, :] = pgas

        m.temp[0, 0] = get_temp(profiles)

        m.vlos[0, 0] = get_vlos(profiles)

        m.vturb[0, 0] = get_vturb(profiles)

        # m.Bln[0,0,0,:] = 100.

        # m.Bho[0,0,0,:] = 100.

        # m.azi[0,0,0,:] = 100. * 3.14159 / 180.

        m.write(
            'wholedata_rps_initial_atmos_{}_{}.nc'.format(
                name,
                '_'.join(
                    [
                        str(profile) for profile in list(profiles)
                    ]
                )
            )
        )
