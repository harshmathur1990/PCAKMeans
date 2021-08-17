import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
from prepare_data import *
from pathlib import Path
import numpy as np
import h5py
import sunpy.io
from helita.io.lp import *
import matplotlib.gridspec as gridspec
from skimage.measure import regionprops


base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
all_data_inversion_rps = new_kmeans / 'all_data_inversion_rps'
selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
mask_file_crisp = base_path / 'crisp_chromis_mask_2019-06-06.fits'
input_file_3950 = '/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/home/harsh/OsloAnalysis/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/home/harsh/OsloAnalysis/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'


new_quiet_profiles = np.array([0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63])
new_shock_profiles = np.array([2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98])
shocks_78_18 = np.array([78, 18])
new_reverse_shock_profiles = np.array([3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97])
new_other_emission_profiles = np.array([5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93])


mask, _  = sunpy.io.fits.read(mask_file_crisp, memmap=True)[0]
mask = np.transpose(mask, axes=(2, 1, 0))

#3950, 6173, 8542
cont = [2.4434714e-05, 4.014861e-08, 4.2277254e-08]

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

wave_indice = np.array(
    [
        4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
        17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
        30,  31,  32,  36,  41,  56,  62,  66,  70,  74,  77,  79,  81,
        83,  85,  87,  89,  92,  96, 100, 104, 110, 125, 143, 150, 158,
       166, 170, 174, 178, 182, 186, 190, 194, 198, 206, 214, 250
    ]
)

photosphere_tau = np.array([-1, 0])


def get_filepath_and_content_list(rp):
    if rp in new_quiet_profiles:
        filename, content_list = all_data_inversion_rps / 'quiet/plots_v1/wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63_cycle_1_t_6_vl_3_vt_4_atmos.nc', [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]

    elif rp in new_shock_profiles:
        filename, content_list = all_data_inversion_rps / 'shocks/plots_v1/wholedata_rps_shock_profiles_rps_2_4_10_19_26_30_37_52_79_85_94_1_22_23_53_55_56_66_67_72_77_80_81_92_87_99_36_6_49_17_96_98_cycle_1_t_6_vl_5_vt_4_atmos.nc', [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]

    elif rp in shocks_78_18:
        filename, content_list = all_data_inversion_rps / 'shocks_78_18/plots_v1/wholedata_rps_shocks_78_18_profile_rps_78_18_cycle_1_t_5_vl_5_vt_4_atmos.nc', [78, 18]

    elif rp in new_reverse_shock_profiles:
        filename, content_list = all_data_inversion_rps / 'reverse/plots_v1/wholedata_rps_reverse_shock_profiles_rps_3_13_16_25_32_33_35_41_45_46_58_61_64_68_82_95_97_cycle_1_t_6_vl_5_vt_4_atmos.nc', [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]

    elif rp in new_other_emission_profiles:
        filename, content_list = all_data_inversion_rps / 'other/plots_v1/wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93_cycle_1_t_6_vl_5_vt_4_atmos.nc', [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]

    return filename, content_list


def get_filepath_and_content_list_input_files(rp):
    if rp in new_quiet_profiles:
        filename, content_list = all_data_inversion_rps / 'wholedata_rps_quiet_profiles_rps_0_11_14_15_20_21_24_28_31_34_40_42_43_47_48_51_60_62_69_70_73_74_75_86_89_90_84_8_44_63.nc', [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]

    elif rp in new_shock_profiles:
        filename, content_list = all_data_inversion_rps / 'wholedata_rps_shock_profiles_rps_2_4_10_19_26_30_37_52_79_85_94_1_22_23_53_55_56_66_67_72_77_80_81_92_87_99_36_6_49_17_96_98.nc', [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98]

    elif rp in shocks_78_18:
        filename, content_list = all_data_inversion_rps / 'wholedata_rps_shocks_78_18_profile_rps_78_18.nc', [78, 18]

    elif rp in new_reverse_shock_profiles:
        filename, content_list = all_data_inversion_rps / 'wholedata_rps_reverse_shock_profiles_rps_3_13_16_25_32_33_35_41_45_46_58_61_64_68_82_95_97.nc', [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]

    elif rp in new_other_emission_profiles:
        filename, content_list = all_data_inversion_rps / 'wholedata_rps_other_emission_profiles_rps_5_7_9_12_27_29_38_39_50_54_57_59_65_71_76_83_88_91_93.nc', [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]

    return filename, content_list


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


def get_input_rps():

    rps = np.zeros((100, 64))

    for i in range(100):
        filename, content_list = get_filepath_and_content_list_input_files(i)

        f = h5py.File(filename, 'r')

        index = content_list.index(i)

        rps[i] = f['profiles'][0, 0, index, wave_indice, 0]

        f.close()

    return rps

def get_mean_calib_velocity():
    _, vlos, _ = get_atmos_values_for_lables()

    atmos_indices = np.where(
        (ltau >= photosphere_tau[0]) &
        (ltau <= photosphere_tau[1])
    )[0]

    vlos = np.mean(vlos[:, atmos_indices], 1)

    f = h5py.File(old_kmeans_file, 'r')

    weights = np.ones(100)

    for i in range(100):

        for frame in selected_frames:
            n, p = np.where(mask[frame] == 1)
            a, = np.where(f['new_final_labels'][frame][n, p] == i)
            weights[i] = a.shape[0]

    f.close()

    return np.sum(vlos * weights) / np.sum(weights)


def classify_rps_in_shocks():
    rps = get_input_rps()

    ca_k_indices = np.arange(0, 29)

    shock_list = list()

    rps = rps[:, ca_k_indices]

    for index, rp in enumerate(rps):
        minima_points = np.r_[True, rp[1:] < rp[:-1]] & np.r_[rp[:-1] < rp[1:], True]

        maxima_points = np.r_[False, rp[1:] > rp[:-1]] & np.r_[rp[:-1] > rp[1:], False]

        minima_indices = np.where(minima_points == True)[0]

        maxima_indices = np.where(maxima_points == True)[0]

        if maxima_indices.size == 1 and minima_indices.size == 2 and maxima_indices[0] <= 15:
            shock_intensity = (rp[maxima_indices[0]] - rp[minima_indices[0]]) / rp[minima_indices[0]]

            shock_list.append((index, shock_intensity))

        if maxima_indices.size == 2 and minima_indices.size == 3 and maxima_indices[0] <= 15:
            if rp[minima_indices[0]] > rp[minima_indices[-1]] and rp[maxima_indices[0]] > rp[maxima_indices[1]]:
                shock_intensity = (rp[maxima_indices[0]] - rp[minima_indices[0]]) / rp[minima_indices[0]]

                shock_list.append((index, shock_intensity))

    return shock_list


def classify_rps_in_reverse_shocks():
    rps = get_input_rps()

    ca_k_indices = np.arange(0, 29)

    shock_list = list()

    rps = rps[:, ca_k_indices]

    for index, rp in enumerate(rps):
        minima_points = np.r_[True, rp[1:] < rp[:-1]] & np.r_[rp[:-1] < rp[1:], True]

        maxima_points = np.r_[False, rp[1:] > rp[:-1]] & np.r_[rp[:-1] > rp[1:], False]

        minima_indices = np.where(minima_points == True)[0]

        maxima_indices = np.where(maxima_points == True)[0]

        if maxima_indices.size == 1 and minima_indices.size == 2 and maxima_indices[0] > 15:
            shock_intensity = (rp[maxima_indices[0]] - rp[minima_indices[1]]) / rp[minima_indices[1]]

            shock_list.append((index, shock_intensity))

        if maxima_indices.size == 2 and minima_indices.size == 3 and maxima_indices[0] <= 15:
            if rp[minima_indices[0]] < rp[minima_indices[2]] and rp[maxima_indices[0]] < rp[maxima_indices[1]]:
                shock_intensity = (rp[maxima_indices[1]] - rp[minima_indices[2]]) / rp[minima_indices[2]]

                shock_list.append((index, shock_intensity))

    return shock_list


def classify_rps_in_other_emissions_and_quiet_profiles():
    rps = get_input_rps()

    ca_k_indices = np.arange(0, 29)

    shock_list = list()

    quiet_profiles = list()

    rps = rps[:, ca_k_indices]

    total_interesting_profiles = list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles) + list(reverse_shocks_profiles)

    for index, rp in enumerate(rps):
        minima_points = np.r_[True, rp[1:] < rp[:-1]] & np.r_[rp[:-1] < rp[1:], True]

        maxima_points = np.r_[False, rp[1:] > rp[:-1]] & np.r_[rp[:-1] > rp[1:], False]

        minima_indices = np.where(minima_points == True)[0]

        maxima_indices = np.where(maxima_points == True)[0]

        if maxima_indices.size > 0:
            if index not in total_interesting_profiles:
                shock_list.append(index)
        else:
            quiet_profiles.append(index)


    return shock_list, quiet_profiles


calib_velocity = -94841.87483891034


# Classification Criteria:
# 1. Emerging Shock Profiles
#    An emission in blue and the difference in the K1 and K2 peaks is < 5%
# 2. Weak Shock Profiles:
#    An emission in blue and the difference in the K1 and K2 peaks is between 5-25%
# 3. Medium Shock Profiles
#    An emission in blue and the difference in the K1 and K2 peaks is between 25-100%
# 4. Strong Shock Profiles
#    An emission in blue and the difference in the K1 and K2 peaks > 100%
# 5. Emerging Reverse Shock Profiles
#    An emission in red and the difference in the K1 and K2 peaks < 5%
# 6. Weak Reverse Shock Profiles
#    An emission in red and the difference in the K1 and K2 peaks is between 5-25%
# 7. Strong Reverse Shock Profiles
#    An emission in red and the difference in the K1 and K2 peaks > 25-52%
# 8. Other Emission Profiles
#    Profiles contaning emission but not in above 4 categories
# 9. Quiet Profiles
#    Profiles without emission

emerging_shock_profiles = np.array(
    [
        76, 40, 7, 71, 2, 81, 53
    ]
)

weak_shocks_profiles = np.array(
    [
        6, 57, 10, 80, 49, 56, 98, 96, 87, 9, 91, 23,  5, 12, 65, 67,
        92
    ]
)

medium_shocks_profiles = np.array(
    [
        1, 55, 39, 22, 94, 30, 54, 93, 17, 77, 26, 72, 52, 19, 79, 37, 4
    ]
)

strong_shocks_profiles = np.array(
    [
        85, 36, 18, 78
    ]
)

emerging_reverse_shock_profiles = np.array(
    [
        41, 83, 64, 33, 3
    ]
)

weak_reverse_shocks_profiles = np.array(
    [
        35, 16, 68, 88, 59, 95, 46
    ]
)

strong_reverese_shock_profiles = np.array(
    [
         97, 25, 13, 45, 32
    ]
)

quiet_profiles = np.array(
    [
        0, 8, 11, 14, 15, 20, 21, 24, 28, 29, 31, 34, 42, 43, 44, 47, 48, 51, 58, 60, 61, 62, 63, 66, 69, 70, 73, 74, 75, 82, 84, 86, 89, 90, 99
    ]
)

other_emission_profiles = np.array(
    [
       27, 38, 50
    ]
)


def categorize_pixels(ref_x, ref_y, del_x, del_y):
    f = h5py.File(old_kmeans_file, 'r')

    categorized = np.zeros(100)

    for i in range(100):
        label_i = f['new_final_labels'][i, ref_x + del_x, ref_y + del_y]
        if label_i in list(quiet_profiles):
            categorized[i] = 0
        elif label_i in list(other_emission_profiles):
            categorized[i] = 1
        elif label_i in list(emerging_shock_profiles):
            categorized[i] = 2
        elif label_i in list(weak_shocks_profiles):
            categorized[i] = 3
        elif label_i in list(medium_shocks_profiles):
            categorized[i] = 4
        elif label_i in list(strong_shocks_profiles):
            categorized[i] = 5
        elif label_i in list(emerging_reverse_shock_profiles):
            categorized[i] = 6
        elif label_i in list(weak_reverse_shocks_profiles):
            categorized[i] = 7
        elif label_i in list(strong_reverese_shock_profiles):
            categorized[i] = 8

    f.close()
    return categorized


def get_shocks_mask(arr):

    mask = np.zeros_like(arr)

    for i in list(weak_shocks_profiles):
        mask[np.where(arr == i)] = 1

    for i in list(medium_shocks_profiles):
        mask[np.where(arr == i)] = 2

    for i in list(strong_shocks_profiles):
        mask[np.where(arr == i)] = 3

    return mask


def get_reverse_shocks_mask(arr):

    mask = np.zeros_like(arr)

    for i in list(weak_reverse_shocks_profiles):
        mask[np.where(arr == i)] = 1

    for i in list(strong_reverese_shock_profiles):
        mask[np.where(arr == i)] = 2

    return mask


def plot_mask(x, y, t):

    plt.close('all')
    plt.clf()
    plt.cla()

    f = h5py.File(old_kmeans_file, 'r')

    shocks_mask = get_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])

    plt.imshow(shocks_mask * mask[t, x:x + 50, y:y + 50], cmap='gray', origin='lower')

    plt.clim(0, 3)

    plt.colorbar()

    plt.show()


def plot_mask_whole(t):

    plt.close('all')
    plt.clf()
    plt.cla()

    f = h5py.File(old_kmeans_file, 'r')

    shocks_mask = get_shocks_mask(f['new_final_labels'][t])

    plt.imshow(shocks_mask * mask[t], cmap='gray', origin='lower')

    plt.clim(0, 3)

    plt.colorbar()

    plt.show()


def plot_reverse_mask(x, y, t):

    plt.close('all')
    plt.clf()
    plt.cla()

    f = h5py.File(old_kmeans_file, 'r')

    mask = get_reverse_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])

    plt.imshow(mask, cmap='gray', origin='lower')

    plt.clim(0, 2)

    plt.colorbar()

    plt.show()


def get_input_profiles(ref_x, ref_y, get_6173=False, get_8542=False):
    whole_data = np.zeros((100, 50, 50, 64))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[:, :, :, 0:30] = np.transpose(
        data[:, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
        axes=(0, 2, 3, 1)
    ) / cont[0]

    if get_6173:
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

        whole_data[:, :, :, 30:30 + 14] = np.transpose(
            data[:, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
            axes=(0, 2, 3, 1)
        ) / cont[1]

    if get_8542:
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

        whole_data[:, :, :, 30 + 14:30 + 14 + 20] = np.transpose(
            data[:, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
            axes=(0, 2, 3, 1)
        ) / cont[2]

    return whole_data


def plot_profile(ref_x, ref_y, x, y, t):
    whole_data = get_input_profiles(ref_x, ref_y)

    plt.plot(
        get_relative_velocity(wave_3933[:-1]),
        whole_data[t, x, y, 0:29]
    )

    plt.show()


def get_doppler_velocity(wavelength, center_wavelength):
    return (wavelength - center_wavelength) * 2.99792458e5 / center_wavelength


@np.vectorize
def get_doppler_velocity_3950(wavelength):
    return get_doppler_velocity(wavelength, 3933.682)


@np.vectorize
def get_relative_velocity(wavelength):
    return wavelength - 3933.682


def plot_lambda_t_curve(ref_x, ref_y, x, y):
    whole_data = get_input_profiles(ref_x, ref_y)
    dv = get_relative_velocity(wave_3933)
    time = np.arange(0, 8.26 * 100, 8.26)

    plt.close('all')
    plt.clf()
    plt.cla()

    X, Y = np.meshgrid(dv[:-1], time)
    plt.pcolormesh(
        X, Y,
        whole_data[:, x, y, 0:29],
        shading='nearest',
        cmap='gray'
    )

    plt.xlabel(r'$\lambda\;(\AA)$')
    plt.ylabel(r'$time\;(seconds)$')

    fig = plt.gcf()

    fig.set_size_inches(4.135, 4.135, forward=True)

    fig.tight_layout()

    plt.savefig(
        'lambda_t_{}_{}_{}_{}.eps'.format(
            ref_x, ref_y, x, y
        ),
        dpi=300,
        format='eps'
    )


def plot_category(x, y, del_x, del_y):

    plt.close('all')
    plt.clf()
    plt.cla()

    category = categorize_pixels(x, y, del_x, del_y)

    plt.scatter(range(100), category)

    plt.show()


def get_rbe_rre_mask(x, y, t):
    f = h5py.File(old_kmeans_file, 'r')
    shocks_mask = get_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])
    reverse_shocks_mask = get_reverse_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])
    f.close()
    new_shocks_mask = np.zeros(
        (3, shocks_mask.shape[0], shocks_mask.shape[1])
    )
    new_reverse_shocks_mask = np.zeros(
        (2, reverse_shocks_mask.shape[0], reverse_shocks_mask.shape[1])
    )
    new_shocks_mask[0][np.where(shocks_mask == 1)] = 1
    new_shocks_mask[1][np.where(shocks_mask == 2)] = 1
    new_shocks_mask[2][np.where(shocks_mask == 3)] = 1
    new_reverse_shocks_mask[0][np.where(reverse_shocks_mask == 1)] = 1
    new_reverse_shocks_mask[1][np.where(reverse_shocks_mask == 2)] = 1
    return new_shocks_mask, new_reverse_shocks_mask


# def get_max_area_region(mask):
#     regionproperties = regionprops(mask)
#     new_mask = np.zeros_like(mask)
#     max_area = -1
#     for a_regionprop in regionproperties:
#         if a_regionprop.area > max_area:
#             max_area = a_regionprop.area
            


def plot_evolution_diagram(
    x,
    y,
    time_indice,
    wave_indice,
    letter=None,
    log_scale=False,
    exp_scale=False,
    shocks_mask=False,
    reverse_shocks_mask=False
):
    whole_data = get_input_profiles(x, y)
    time = np.arange(0, 8.26 * 100, 8.26)
    time = np.round(time, 2)
    labels = None
    if shocks_mask or reverse_shocks_mask:
        f = h5py.File(old_kmeans_file, 'r')
        labels = f['new_final_labels'][time_indice][:, x:x + 50, y:y + 50]
        f.close()
    min_value = whole_data[time_indice][:, :, :, wave_indice].min()
    max_value = whole_data[time_indice][:, :, :, wave_indice].max()

    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.figure(
        figsize=(
            4.135,
            4.135 * time_indice.size / wave_indice.size
        )
    )
    gs1 = gridspec.GridSpec(time_indice.size, wave_indice.size)
    gs1.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    k = 0
    for index_t, t in enumerate(time_indice):
        for index_w, w in enumerate(wave_indice):
            dv = np.round(
                get_doppler_velocity_3950(wave_3933[w]),
                1
            )
            axs = plt.subplot(gs1[k])
            if log_scale:
                im = axs.imshow(
                    np.log(whole_data[t, :, :, w]),
                    cmap='gray',
                    origin='lower'
                )
                im.set_clim(np.log(min_value), np.log(max_value))
            elif exp_scale:
                im = axs.imshow(
                    np.exp(whole_data[t, :, :, w]),
                    cmap='gray',
                    origin='lower'
                )
                im.set_clim(np.exp(min_value), np.exp(max_value))
            else:
                im = axs.imshow(
                    whole_data[t, :, :, w],
                    cmap='gray',
                    origin='lower'
                )
                im.set_clim(min_value, max_value)
            if index_t == 0:
                axs.text(
                    0.1, 0.8, r'${}\;km/sec$'.format(dv),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='xx-small'
                )
            if index_w == 0:
                axs.text(
                    0.1, 0.6, r'${}\;s$'.format(time[t]),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='xx-small'
                )
            if index_t == 1 and index_w == wave_indice.size-1:
                axs.text(
                    0.1, 0.6, r'{}'.format(letter),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='xx-small'
                )

            axs.spines["top"].set_color("white")
            axs.spines["left"].set_color("white")
            axs.spines["bottom"].set_color("white")
            axs.spines["right"].set_color("white")

            axs.spines["top"].set_linewidth(0.1)
            axs.spines["left"].set_linewidth(0.1)
            axs.spines["bottom"].set_linewidth(0.1)
            axs.spines["right"].set_linewidth(0.1)

            axs.tick_params(axis='x', colors='white')
            axs.tick_params(axis='y', colors='white')

            mask = None
            color = None
            if shocks_mask:
                mask = get_shocks_mask(labels[t])
                mask[np.where(mask >= 1)] = 1
                color='#FF4848'
                axs.contour(
                    mask,
                    origin='lower',
                    colors=color,
                    linewidths=0.2,
                    alpha=0.2
                )
            if reverse_shocks_mask:
                mask = get_reverse_shocks_mask(labels[t])
                mask[np.where(mask >= 1)] = 1
                color='#0F52BA'
                axs.contour(
                    mask,
                    origin='lower',
                    colors=color,
                    linewidths=0.2,
                    alpha=0.2
                )

            axs.set_xticklabels([])
            axs.set_yticklabels([])
            k += 1

    # fig.tight_layout()
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        'fov_evolution_{}_{}_t_{}_w_{}.eps'.format(
            x, y, '_'.join([str(a) for a in time_indice]), '_'.join([str(a) for a in wave_indice])
        ),
        dpi=300,
        format='eps'
    )
    fig.savefig(
        'fov_evolution_{}_{}_t_{}_w_{}.pdf'.format(
            x, y, '_'.join([str(a) for a in time_indice]), '_'.join([str(a) for a in wave_indice])
        ),
        dpi=300,
        format='pdf'
    )

    plt.close('all')
    plt.clf()
    plt.cla()


def plot_2_profiles(ref_x, ref_y, x1, y1, t1, x2, y2, t2):
    whole_data = get_input_profiles(ref_x, ref_y)

    plt.plot(
        get_doppler_velocity_3950(wave_3933[:-1]),
        whole_data[t1, x1, y1, 0:29], color='#FF4848'
    )

    plt.plot(
        get_doppler_velocity_3950(wave_3933[:-1]), 
        whole_data[t2, x2, y2, 0:29], color='#0F52BA'
    )

    plt.xlabel(r'$\Delta\;(kms^{-1})$')

    plt.ylabel(r'$I/I_{c}$')

    fig = plt.gcf()

    fig.set_size_inches(5.875, 4.125, forward=True)

    fig.tight_layout()

    fig.savefig(
        'plot_profiles_ref_x_{}_ref_y_{}_x1_{}_y1_{}_t1_{}_x2_{}_y2_{}_t2_{}.eps'.format(
            ref_x, ref_y, x1, y1, t1, x2, y2, t2
        ),
        dpi=300,
        format='eps'
    )

    plt.close('all')
    plt.clf()
    plt.cla()


def plot_1_profile_maps(ref_x, ref_y, x, y, t, nrow, ncol):

    whole_data = get_input_profiles(ref_x, ref_y)
    time = np.arange(0, 8.26 * 100, 8.26)
    time = np.round(time, 2)
    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.figure(
        figsize=(
            4.135,
            4.135 * nrow / ncol
        )
    )
    gs1 = gridspec.GridSpec(nrow, ncol)
    gs1.update(wspace=0.0, hspace=0.0)

    k = 0
    for index_t in range(nrow):
        for index_w in range(ncol):
            axs = plt.subplot(gs1[k])
            im, = axs.plot(
                get_doppler_velocity_3950(wave_3933[:-1]),
                whole_data[t[k], x, y, 0:29]
            )
            if index_t == 0:
                axs.text(
                    0.1, 0.8, r'${}\;km/sec$'.format(dv),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='xx-small'
                )
            if index_w == 0:
                axs.text(
                    0.1, 0.6, r'${}\;s$'.format(time[t]),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='xx-small'
                )

            mask = None
            color = None
            if shocks_mask:
                mask = get_shocks_mask(labels[t])
                mask[np.where(mask >= 1)] = 1
                color='#FF4848'
                axs.contour(
                    mask,
                    origin='lower',
                    colors=color,
                    linewidths=0.2,
                    alpha=0.2
                )
            if reverse_shocks_mask:
                mask = get_reverse_shocks_mask(labels[t])
                mask[np.where(mask >= 1)] = 1
                color='#0F52BA'
                axs.contour(
                    mask,
                    origin='lower',
                    colors=color,
                    linewidths=0.2,
                    alpha=0.2
                )

            axs.set_xticklabels([])
            axs.set_yticklabels([])
            k += 1

    # fig.tight_layout()
    fig.savefig(
        'fov_evolution_{}_{}_t_{}_w_{}.eps'.format(
            x, y, '_'.join([str(a) for a in time_indice]), '_'.join([str(a) for a in wave_indice])
        ),
        dpi=300,
        format='eps'#,
        # bbox_inches='tight'
    )

    plt.close('all')
    plt.clf()
    plt.cla()

# TODO:
#
# 1. Define fovs with timesteps such that 10 examples
#    of each type of shocks and reverse shocks
# 2. Write Paper