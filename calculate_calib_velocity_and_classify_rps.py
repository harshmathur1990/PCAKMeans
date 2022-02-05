import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
from prepare_data import *
from pathlib import Path
import numpy as np
import h5py
import sunpy.io
from helita.io.lp import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.measure import regionprops
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
response_functions_path = new_kmeans / 'Response Functions'
all_data_inversion_rps = new_kmeans / 'all_data_inversion_rps'
selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
mask_file_crisp = base_path / 'crisp_chromis_mask_2019-06-06.fits'
input_file_3950 = '/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/home/harsh/OsloAnalysis/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/home/harsh/OsloAnalysis/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173_blos = Path('/home/harsh/OsloAnalysis/Blos.6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')
input_file_hmi_blos = Path('/home/harsh/OsloAnalysis/hmimag_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.icube')
input_87_17_14_file = response_functions_path / 'wholedata_rps_87_17_14.nc'
response_87_17_14_file = response_functions_path / 'wholedata_rps_87_17_14_result_atmos_response.nc'
output_atmos_87_17_14_file = response_functions_path / 'wholedata_rps_87_17_14_result_atmos.nc'
input_78_18_file = response_functions_path / 'wholedata_rps_shocks_78_18_profile_rps_78_18.nc'
response_78_18_file = response_functions_path / 'wholedata_rps_shocks_78_18_profile_rps_78_18_cycle_1_t_5_vl_5_vt_4_response.nc'
response_662_file = new_kmeans / 'wholedata_inversions/fov_662_712_708_758/plots/wholedata_x_662_712_y_708_758_ref_x_25_ref_y_18_t_4_11_output_model_response.nc'
output_atmos_78_18_file = response_functions_path / 'wholedata_rps_shocks_78_18_profile_rps_78_18_cycle_1_t_5_vl_5_vt_4_atmos.nc'

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


def reclassify_rps_in_shocks():
    candidate_shock_rps = list(emerging_shock_profiles) + list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles)

    temp, vlos, vturb = get_atmos_values_for_lables()

    vlos -= calib_velocity

    vlos /= 1e5

    vturb /= 1e5

    new_shock_rps = list()

    for cp in candidate_shock_rps:
        if vlos[cp, 112] < 0:
            new_shock_rps.append(cp)

    return new_shock_rps


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


calib_velocity = -94841.87483891034 -18000


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

very_strong_shocks_profiles = np.array(
    [
        18, 78
    ]
)

very_very_strong_shocks_profiles = np.array(
    [
        78
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

    for i in list(strong_shocks_profiles):
        mask[np.where(arr == i)] = 1

    return mask


def get_very_strong_shocks_mask(arr):

    mask = np.zeros_like(arr)

    for i in list(very_strong_shocks_profiles):
        mask[np.where(arr == i)] = 1

    return mask


def get_very_very_strong_shocks_mask(arr):

    mask = np.zeros_like(arr)

    for i in list(very_very_strong_shocks_profiles):
        mask[np.where(arr == i)] = 1

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

    shocks_mask = get_very_strong_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])

    plt.imshow(
        np.multiply(
            shocks_mask,
            mask[t, x:x + 50, y:y + 50]
        ),
        cmap='gray',
        origin='lower'
    )

    plt.clim(0, 1)

    plt.colorbar()

    plt.show()


def plot_very_strong_mask(x, y, t):

    plt.close('all')
    plt.clf()
    plt.cla()

    f = h5py.File(old_kmeans_file, 'r')

    shocks_mask = get_very_strong_shocks_mask(f['new_final_labels'][t, x:x + 50, y:y + 50])

    plt.imshow(
        np.multiply(
            shocks_mask,
            mask[t, x:x + 50, y:y + 50]
        ),
        cmap='gray',
        origin='lower'
    )

    plt.clim(0, 1)

    plt.colorbar()

    plt.show()

def plot_mask_all(t):

    plt.close('all')
    plt.clf()
    plt.cla()

    f = h5py.File(old_kmeans_file, 'r')

    shocks_mask = get_shocks_mask(f['new_final_labels'][t])

    plt.imshow(np.multiply(shocks_mask, mask[t]), cmap='gray', origin='lower')

    plt.clim(0, 1)

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


def get_input_profiles(ref_x, ref_y, get_6173=False, get_8542=False, get_6173_blos=False, time_step=None):

    if time_step is None:
        time_step = np.arange(100)

    whole_data = np.zeros((100, 50, 50, 64))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[time_step, :, :, 0:30] = np.transpose(
        data[time_step, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
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

        whole_data[time_step, :, :, 30:30 + 14] = np.transpose(
            data[time_step, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
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

        whole_data[time_step, :, :, 30 + 14:30 + 14 + 20] = np.transpose(
            data[time_step, 0, :, ref_x: ref_x + 50, ref_y: ref_y + 50],
            axes=(0, 2, 3, 1)
        ) / cont[2]

    if get_6173_blos:

        sh, dt, header = getheader(input_file_6173_blos)

        data = np.memmap(
            input_file_6173_blos,
            mode='r',
            shape=sh,
            dtype=dt,
            order='F',
            offset=512
        )

        blos_6173 = np.transpose(
            data,
            axes=(2, 1, 0)
        )

        return whole_data, blos_6173[:, ref_x:ref_x + 50, ref_y:ref_y + 50]

    return whole_data


def plot_new_evolution_diagram(ref_x, ref_y, time_step, wave_indice, mark_t, mark_x, mark_y, letter, blos_lim=30):

    write_path = Path(
        '/home/harsh/Shocks Paper/Shocks Evolution Plots/'
    )

    fontsize = 8

    whole_data, blos_6173 = get_input_profiles(ref_x, ref_y, get_6173_blos=True, time_step=time_step)

    whole_data = whole_data[time_step]
    blos_6173 = blos_6173[time_step]

    f = h5py.File(old_kmeans_file, 'r')
    labels = f['new_final_labels'][time_step][:, ref_x:ref_x + 50, ref_y:ref_y + 50]
    f.close()

    time_arr = np.round(
        np.arange(0, 826, 8.26),
        1
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(3.5, 7 * 3.5 / 5))

    gs = gridspec.GridSpec(7, 5)

    gs.update(left=0.17, right=1, top=0.9, bottom=0.07, wspace=0.0, hspace=0.0)

    axs = list()

    k = 0

    wb_vmin = whole_data[:, :, :, 29].min()
    wb_vmax = whole_data[:, :, :, 29].max()

    blos_vmin = -1 * blos_lim
    blos_vmax =  blos_lim

    factor = 0.8

    im_vmin = whole_data[:, :, :, wave_indice].min()
    im_vmax = whole_data[:, :, :, wave_indice].max() * factor

    for i in range(7):
        axsi = list()
        for j in range(5):
            axsi.append(fig.add_subplot(gs[k]))

            k += 1
        axs.append(axsi)

    for i in range(7):
        for j in range(5):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].set_xticklabels([])
            axs[i][j].set_yticklabels([])
            if i == 6:
                axs[i][j].set_xticks([int(50 / 1.85)])
                axs[i][j].xaxis.set_minor_locator(MultipleLocator(25/1.85))
            if j == 0:
                axs[i][j].set_yticks([int(50 / 1.85)])
                axs[i][j].yaxis.set_minor_locator(MultipleLocator(25 / 1.85))
            if i == 6 and j == 0:
                axs[i][j].set_xticklabels([1])
                axs[i][j].text(
                    0.0, -0.55,
                    r'$\mathrm{\Delta x}$ [arcsec]',
                    transform=axs[i][j].transAxes,
                    fontsize=fontsize
                )
                axs[i][j].set_yticklabels([1])
                axs[i][j].set_ylabel(r'$\mathrm{\Delta y}$ [arcsec]', fontsize=fontsize)
            if j == 0:
                axs[i][j].imshow(
                    whole_data[i, :, :, 29],
                    cmap='gray',
                    origin='lower',
                    vmin=wb_vmin,
                    vmax=wb_vmax
                )
                if i == 0:
                    axs[i][j].text(
                        -0.4, 1.1,
                        r'(a)',
                        transform=axs[i][j].transAxes,
                        fontsize=fontsize + 2
                    )
                    axs[i][j].text(
                        0.15, 1.1,
                        r'$\mathrm{4000\;\AA}$',
                        transform=axs[i][j].transAxes,
                        fontsize=fontsize
                    )

            elif j == 1:
                im = axs[i][j].imshow(
                    blos_6173[i, :, :],
                    cmap='gray',
                    origin='lower',
                    vmin=blos_vmin,
                    vmax=blos_vmax
                )

                if i == 0:
                    axs[i][j].set_title(
                        r'$B_{\mathrm{LOS}}$',
                        fontsize=fontsize
                    )

                    cbaxes = inset_axes(
                        axs[i][j],
                        width="70%",
                        height="10%",
                        loc=3,
                        # borderpad=5
                    )
                    cbar = fig.colorbar(
                        im,
                        cax=cbaxes,
                        ticks=[blos_vmin, blos_vmax],
                        orientation='horizontal'
                    )

                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.tick_params(colors='white', labelsize=fontsize)

            else:
                axs[i][j].imshow(
                    whole_data[i, :, :, wave_indice[j - 2]],
                    cmap='gray',
                    origin='lower',
                    vmin=im_vmin,
                    vmax=im_vmax
                )
                if i == 0:
                    axs[i][j].set_title(
                        r'${0:+.1f}\mathrm{{\;m\AA}}$'.format(
                            np.round(
                                get_relative_velocity(
                                    wave_3933[wave_indice[j - 2]]
                                ) * 1000,
                                1
                            )
                        ),
                        fontsize=fontsize - 1
                    )
                if j == 4:
                    time_color = 'white'
                    axs[i][j].text(
                        0.3, 0.05,
                        r'{} s'.format(
                            time_arr[time_step[i]]
                        ),
                        transform=axs[i][j].transAxes,
                        fontsize=fontsize,
                        color=time_color,
                        rotation=0,
                        # labelpad=20
                    )

            size = plt.rcParams['lines.markersize']
            if mark_t == time_step[i] and j == 2:
                axs[i][j].scatter(
                    mark_y, mark_x,
                    marker='+',
                    color='red',
                    linewidths=1,
                    s=(size**2) * 8
                )

            lightblue = '#5089C6'
            mediumdarkblue = '#035397'
            darkblue = '#001E6C'
            mask = get_shocks_mask(labels[i])
            mask[np.where(mask >= 1)] = 1
            axs[i][j].contour(
                mask,
                origin='lower',
                colors=lightblue,
                linewidths=1,
                alpha=1,
                levels=0
            )
            mask = get_very_strong_shocks_mask(labels[i])
            mask[np.where(mask >= 1)] = 1
            axs[i][j].contour(
                mask,
                origin='lower',
                colors=mediumdarkblue,
                linewidths=1,
                alpha=1,
                levels=0
            )
            mask = get_very_very_strong_shocks_mask(labels[i])
            mask[np.where(mask >= 1)] = 1
            axs[i][j].contour(
                mask,
                origin='lower',
                colors=darkblue,
                linewidths=1,
                alpha=1,
                levels=0
            )

    fig.savefig(
        write_path / 'FoV_{}.pdf'.format(
            letter
        ),
        dpi=300,
        format='pdf'
    )
    fig.savefig(
        write_path / 'FoV_{}.png'.format(
            letter
        ),
        dpi=300,
        format='png'
    )
    plt.close('all')

    plt.clf()

    plt.cla()


def make_nb_image(ref_x, ref_y, time_step, wave_indice, mark_x, mark_y, letter):
    write_path = Path(
        '/home/harsh/Shocks Paper/Shocks Evolution Plots/'
    )

    whole_data = get_input_profiles(ref_x, ref_y, time_step=time_step)

    whole_data = whole_data[time_step][0]

    f = h5py.File(old_kmeans_file, 'r')
    labels = f['new_final_labels'][time_step][0][ref_x:ref_x + 50, ref_y:ref_y + 50]
    f.close()

    time_arr = np.round(
        np.arange(0, 826, 8.26),
        1
    )

    fontsize = 8
    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(1.4, 1.6))

    gs = gridspec.GridSpec(1, 1)

    gs.update(left=0, right=1, top=0.875, bottom=0, wspace=0.0, hspace=0.0)

    axs = fig.add_subplot(gs[0])

    axs.imshow(whole_data[:, :, wave_indice], cmap='gray', origin='lower')

    axs.text(
        0.4, 1.08,
        r'FoV {}'.format(
            letter
        ),
        transform=axs.transAxes,
        color='black',
        fontsize=fontsize
    )

    axs.text(
        0.15, 1.005,
        r'Ca II K ${}$ m$\mathrm{{\AA}}$'.format(
            np.round(
                get_relative_velocity(
                    wave_3933[wave_indice]
                ) * 1000,
                1
            )
        ),
        transform=axs.transAxes,
        color='black',
        fontsize=fontsize
    )

    axs.scatter(
        mark_y, mark_x,
        marker='+',
        color='red',
        linewidths=0.8
    )

    lightblue = '#5089C6'
    mediumdarkblue = '#035397'
    darkblue = '#001E6C'
    mask = get_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    axs.contour(
        mask,
        origin='lower',
        colors=lightblue,
        linewidths=1,
        alpha=1,
        levels=0
    )
    mask = get_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    axs.contour(
        mask,
        origin='lower',
        colors=mediumdarkblue,
        linewidths=1,
        alpha=1,
        levels=0
    )
    mask = get_very_very_strong_shocks_mask(labels)
    mask[np.where(mask >= 1)] = 1
    axs.contour(
        mask,
        origin='lower',
        colors=darkblue,
        linewidths=0.5,
        alpha=1,
        levels=0
    )

    axs.set_xticklabels([])
    axs.set_yticklabels([])

    fig.savefig(
        write_path / 'FoV_one_t_step_{}.pdf'.format(
            letter
        ),
        dpi=300,
        format='pdf'
    )
    fig.savefig(
        write_path / 'FoV_one_t_step_{}.png'.format(
            letter
        ),
        dpi=300,
        format='png'
    )
    plt.close('all')

    plt.clf()

    plt.cla()


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


@np.vectorize
def get_relative_velocity_8542(wavelength):
    return wavelength - 8542.09


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
    reverse_shocks_mask=False,
    mark_x=None, mark_y=None, mark_t=None
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
                get_relative_velocity(wave_3933[w]),
                2
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
            if mark_t == t and index_w == 2:
                axs.scatter(
                    mark_y, mark_x,
                    marker='+',
                    color='blue'
                )
            if index_t == 0:
                axs.text(
                    0.1, 0.8, r'${}\;\AA$'.format(dv),
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
                    0.1, 0.6, '{}'.format(letter),
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
                mask = get_shocks_mask(labels[index_t])
                mask[np.where(mask >= 1)] = 1
                color='blue'
                axs.contour(
                    mask,
                    origin='lower',
                    colors=color,
                    linewidths=0.2,
                    alpha=0.2
                )
            if reverse_shocks_mask:
                mask = get_reverse_shocks_mask(labels[index_t])
                mask[np.where(mask >= 1)] = 1
                color='red'
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



def plot_1_profiles(ref_x, ref_y, x1, y1, t1):
    whole_data = get_input_profiles(ref_x, ref_y)

    plt.plot(
        get_doppler_velocity_3950(wave_3933[:-1]),
        whole_data[t1, x1, y1, 0:29], color='#0F52BA'
    )

    plt.xlabel(r'$\Delta\;(kms^{-1})$')

    plt.ylabel(r'$I/I_{c}$')

    fig = plt.gcf()

    fig.set_size_inches(5.875, 4.125, forward=True)

    fig.tight_layout()

    fig.savefig(
        'plot_profiles_ref_x_{}_ref_y_{}_x1_{}_y1_{}_t1_{}.pdf'.format(
            ref_x, ref_y, x1, y1, t1
        ),
        dpi=300,
        format='pdf'
    )

    plt.close('all')
    plt.clf()
    plt.cla()


def make_evolution_single_pixel_plot(ref_x, ref_y, x, y, time_indice_1, time_indice_2, letter, wave_indice, color_list_1=None, color_list_2=None):
    write_path = Path(
        '/home/harsh/Shocks Paper/Shocks Evolution Plots/'
    )

    fontsize = 8

    whole_data = get_input_profiles(ref_x, ref_y, get_8542=True, time_step=time_indice_1)
    time = np.round(
        np.arange(0, 8.26*100, 8.26),
        2
    )

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(1.75, 1.4))
    plt.subplots_adjust(left=0.28, right=0.99, bottom=0.27, top=0.98)
    for index, t in enumerate(time_indice_1):
        if color_list_1 is not None:
            axs.plot(
                get_relative_velocity(wave_3933[:-1]),
                whole_data[t, x, y, 0:29],
                color=color_list_1[index]
            )
        else:
            axs.plot(
                get_relative_velocity(wave_3933[:-1]),
                whole_data[t, x, y, 0:29],
                label=r'$\mathrm{{t={}s}}$'.format(
                    time[t]
                )
            )

    for wi in wave_indice:
        axs.axvline(
            get_relative_velocity(wave_3933[wi]),
            linestyle='--',
            color='black',
            linewidth=0.5
        )

    if color_list_1 is None:
        axs.legend(loc="upper right")

    axs.text(
        0.05, 0.85,
        r'(d)',
        transform=axs.transAxes,
        color='black',
        fontsize=fontsize + 2
    )
    axs.set_xticks([-0.5, 0, 0.5])

    axs.set_xticklabels([-0.5, 0, 0.5])

    axs.xaxis.set_tick_params(labelsize=fontsize)

    axs.yaxis.set_tick_params(labelsize=fontsize)

    axs.text(
        0.36, -0.35,
        r'$\Delta \lambda\;\mathrm{[\AA]}$',
        transform=axs.transAxes,
        fontsize=fontsize
    )

    axs.set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)

    # axs.set_title(
    #     'FoV {}'.format(
    #         letter
    #     )
    # )

    # fig.tight_layout()

    fig.savefig(
        write_path / 'pixel_evolution_CaK_{}.pdf'.format(
            letter
        ),
        format='pdf',
        dpi=300
    )

    fig.savefig(
        write_path / 'pixel_evolution_CaK_{}.png'.format(
            letter
        ),
        format='png',
        dpi=300
    )

    plt.close('all')
    plt.clf()
    plt.cla()

    whole_data = get_input_profiles(ref_x, ref_y, get_8542=True, time_step=time_indice_2)
    fig, axs = plt.subplots(1, 1, figsize=(1.75, 1.4))
    plt.subplots_adjust(left=0.28, right=0.99, bottom=0.27, top=0.98)

    for index, t in enumerate(time_indice_2):
        if color_list_2 is not None:
            axs.plot(
                get_relative_velocity_8542(wave_8542),
                whole_data[t, x, y, 30 + 14:30 + 14 + 20],
                color=color_list_2[index]
            )
        else:
            axs.plot(
                get_relative_velocity_8542(wave_8542),
                whole_data[t, x, y, 30 + 14:30 + 14 + 20],
                label=r'$t={}s$'.format(
                    time[t]
                )
            )

    if color_list_2 is None:
        axs.legend(loc="upper right")

    axs.text(
        0.05, 0.85,
        r'(e)',
        transform=axs.transAxes,
        color='black',
        fontsize=fontsize + 2
    )

    axs.set_xticks([-1, 0, 1])

    axs.set_xticklabels([-1, 0, 1])

    axs.xaxis.set_tick_params(labelsize=fontsize)

    axs.yaxis.set_tick_params(labelsize=fontsize)

    axs.text(
        0.36, -0.35,
        r'$\Delta \lambda\;\mathrm{[\AA]}$',
        transform=axs.transAxes,
        fontsize=fontsize
    )

    axs.set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)

    # axs.set_title(
    #     'FoV {}'.format(
    #         letter
    #     )
    # )

    # fig.tight_layout()

    fig.savefig(
        write_path / 'pixel_evolution_CaIR_{}.pdf'.format(
            letter
        ),
        format='pdf',
        dpi=300
    )

    fig.savefig(
        write_path / 'pixel_evolution_CaIR_{}.png'.format(
            letter
        ),
        format='png',
        dpi=300
    )

    plt.close('all')
    plt.clf()
    plt.cla()


def make_lambda_t_curve(ref_x, ref_y, x, y, time_step, mark_t_1, mark_t_2, color_list_1, color_list_2, letter, appendix=False, wave_indice=12):
    write_path = Path(
        '/home/harsh/Shocks Paper/Shocks Evolution Plots/'
    )

    whole_data = get_input_profiles(ref_x, ref_y, get_8542=True, time_step=time_step)
    dv = get_relative_velocity(wave_3933)
    dv8542 = get_relative_velocity_8542(wave_8542)

    time = np.arange(0, 826, 8.26)

    fontsize = 8

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(1.75, 3.5))
    if not appendix:
        plt.subplots_adjust(left=0.28, right=0.99, bottom=0.05, top=0.86)
    else:
        plt.subplots_adjust(left=0.28, right=0.99, bottom=0.13, top=0.9)
    axs.imshow(
        whole_data[time_step, x, y, 0:29],
        extent=[dv[0], dv[-2], time[time_step[0]], time[time_step[-1]]],
        cmap='gray',
        origin='lower',
        aspect='auto'
    )

    for i, tt in enumerate(mark_t_1):
        if not appendix:
            axs.scatter([0], [time[tt]], color=color_list_1[i], marker='+')
        else:

            xx = np.round(
                get_relative_velocity(
                    wave_3933[wave_indice]
                ),
                2
            )
            axs.scatter([xx], [time[tt]], color=color_list_1[i], marker='+')

    if not appendix:
        axs.text(
            0.05, 0.9,
            r'(b)',
            transform=axs.transAxes,
            color='white',
            fontsize=fontsize + 2
        )
    axs.set_xticks([-0.5, 0, 0.5])
    if not appendix:
        axs.set_xticklabels([])
    else:
        axs.set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)
        axs.set_xlabel(r'$\Delta \lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
    axs.yaxis.set_tick_params(labelsize=fontsize)
    axs.set_ylabel(r'time [s]', fontsize=fontsize)
    axs.set_title(
        r'Ca II K',
        fontsize=fontsize
    )

    # fig.tight_layout()

    if not appendix:
        fig.savefig(
            write_path / 'lambda_t_FoV_CaK_{}.pdf'.format(
                letter
            ),
            dpi=300,
            format='pdf',
            # bbox_inches='tight'
        )
        fig.savefig(
            write_path / 'lambda_t_FoV_CaK_{}.png'.format(
                letter
            ),
            dpi=300,
            format='png',
            # bbox_inches='tight'
        )
    else:
        fig.savefig(
            write_path / 'lambda_t_FoV_CaK_{}_appendix.pdf'.format(
                letter
            ),
            dpi=300,
            format='pdf',
            # bbox_inches='tight'
        )

        fig.savefig(
            write_path / 'lambda_t_FoV_CaK_{}_appendix.png'.format(
                letter
            ),
            dpi=300,
            format='png',
            # bbox_inches='tight'
        )
    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(1.75, 3.5))
    if not appendix:
        plt.subplots_adjust(left=0.28, right=0.99, bottom=0.05, top=0.86)
    else:
        plt.subplots_adjust(left=0.28, right=0.99, bottom=0.13, top=0.9)
    axs.imshow(
        whole_data[time_step, x, y, 30 + 14:30 + 14 + 20],
        extent=[dv8542[0], dv8542[-1], time[time_step[0]], time[time_step[-1]]],
        cmap='gray',
        origin='lower',
        aspect='auto'
    )

    for i, tt in enumerate(mark_t_2):
        axs.scatter([0], [time[tt]], color=color_list_2[i], marker='+')

    if not appendix:
        axs.text(
            0.05, 0.9,
            r'(c)',
            transform=axs.transAxes,
            color='white',
            fontsize=fontsize + 2
        )
    axs.set_xticks([-1, 0, 1])
    if not appendix:
        axs.set_xticklabels([])
    else:
        axs.set_xticklabels([-1, 0, 1], fontsize=fontsize)
        axs.set_xlabel(r'$\Delta \lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
    axs.yaxis.set_tick_params(labelsize=fontsize)
    axs.set_ylabel(r'time [s]', fontsize=fontsize)
    axs.set_title(
        r'Ca II 8542 $\mathrm{\AA}$',
        fontsize=fontsize
    )

    # fig.tight_layout()

    if not appendix:
        fig.savefig(
            write_path / 'lambda_t_FoV_CaIR_{}.pdf'.format(
                letter
            ),
            dpi=300,
            format='pdf',
            # bbox_inches='tight'
        )
        fig.savefig(
            write_path / 'lambda_t_FoV_CaIR_{}.png'.format(
                letter
            ),
            dpi=300,
            format='png',
            # bbox_inches='tight'
        )
    else:
        fig.savefig(
            write_path / 'lambda_t_FoV_CaIR_{}_appendix.pdf'.format(
                letter
            ),
            dpi=300,
            format='pdf',
            # bbox_inches='tight'
        )
        fig.savefig(
            write_path / 'lambda_t_FoV_CaIR_{}_appendix.png'.format(
                letter
            ),
            dpi=300,
            format='png',
            # bbox_inches='tight'
        )

    plt.close('all')
    plt.clf()
    plt.cla()


def make_shock_evolution_plots():
    # ref_x_list = [662, 915, 486, 582, 810, 455, 95, 315, 600, 535]
    #
    # ref_y_list = [708, 1072, 974, 627, 335, 940, 600, 855, 1280, 715]

    ref_x_list = [662, 582, 486, 810, 455, 315, 600, 535]

    ref_y_list = [708, 627, 974, 335, 940, 855, 1280, 715]

    time_step_list = [
        np.arange(4, 11),
        # np.arange(14, 21),
        np.arange(32, 39),
        np.arange(17, 24),
        np.arange(12, 19),
        np.arange(57, 64),
        # np.arange(93, 100),
        np.arange(7, 14),
        np.arange(8, 15),
        np.arange(9, 16)
    ]

    time_step_adhoc_list = [
        np.array([3, 5, 6, 7, 8, 9, 12]),
        # np.arange(14, 21),
        np.array([30, 33, 34, 35, 36, 37, 40]),
        np.arange(17, 24),
        np.array([11, 14, 15, 16, 17, 18, 21]),
        np.arange(57, 64),
        # np.arange(93, 100),
        np.array([6, 10, 11, 12, 13, 15, 16]),
        np.array([6, 10, 11, 12, 13, 14, 17]),
        np.array([8, 11, 12, 13, 14, 15, 17])
    ]

    mark_list = [
        [6 , 25, 18],
        # [16, 16, 15],
        [35, 29, 20],
        [20, 23, 27],
        [15, 24, 22],
        [59, 28, 28],
        # [97, 24, 22],
        [11, 22, 16],
        [11, 26, 21],
        [12, 28, 28]
    ]

    FoV_letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  #, 'I', 'J']

    wave_indice = np.array([12, 14, 16])

    blos_lim = 100

    color_list_2 = ['blue', 'orange', 'red']

    color_list_1 = ['blue', 'green', 'orange', 'brown', 'red']

    for i in range(8):
        plot_new_evolution_diagram(
            ref_x_list[i],
            ref_y_list[i],
            time_step_adhoc_list[i],
            wave_indice,
            mark_list[i][0],
            mark_list[i][1],
            mark_list[i][2],
            FoV_letter_list[i],
            blos_lim
        )

        # make_evolution_single_pixel_plot(
        #     ref_x_list[i],
        #     ref_y_list[i],
        #     mark_list[i][1],
        #     mark_list[i][2],
        #     np.array(
        #         [
        #             mark_list[i][0] - 4 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0] - 2 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0],
        #             mark_list[i][0] + 2 if (mark_list[i][0] + 4) < 100 else 99,
        #             mark_list[i][0] + 4 if (mark_list[i][0] + 4) < 100 else 99
        #         ]
        #     ),
        #     np.array(
        #         [
        #             mark_list[i][0] - 4 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0],
        #             mark_list[i][0] + 4 if (mark_list[i][0] + 4) < 100 else 99
        #         ]
        #     ),
        #     FoV_letter_list[i],
        #     wave_indice,
        #     color_list_1,
        #     color_list_2
        # )
        #
        # start_t = time_step_list[i][0]
        # end_t = time_step_list[i][-1]
        #
        # begin_seq = np.arange(start_t-14 if (start_t-14) >= 0 else 0, start_t)
        # end_seq = np.arange(end_t, end_t + 14 if (end_t + 14) < 100 else 99)
        #
        # time_step = np.array(list(begin_seq) + list(time_step_list[i]) + list(end_seq))
        # make_lambda_t_curve(
        #     ref_x_list[i],
        #     ref_y_list[i],
        #     mark_list[i][1],
        #     mark_list[i][2],
        #     time_step,
        #     np.array(
        #         [
        #             mark_list[i][0] - 4 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0] - 2 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0],
        #             mark_list[i][0] + 2 if (mark_list[i][0] + 4) < 100 else 99,
        #             mark_list[i][0] + 4 if (mark_list[i][0] + 4) < 100 else 99
        #         ]
        #     ),
        #     np.array(
        #         [
        #             mark_list[i][0] - 4 if (mark_list[i][0] - 4) >= 0 else 0,
        #             mark_list[i][0],
        #             mark_list[i][0] + 4 if (mark_list[i][0] + 4) < 100 else 99
        #         ]
        #     ),
        #     color_list_1,
        #     color_list_2,
        #     FoV_letter_list[i]
        # )
        #
        # make_nb_image(
        #     ref_x_list[i],
        #     ref_y_list[i],
        #     np.array([mark_list[i][0]]),
        #     wave_indice[0],
        #     mark_list[i][1],
        #     mark_list[i][2],
        #     FoV_letter_list[i]
        # )
        #
        # make_lambda_t_curve(
        #     ref_x_list[i],
        #     ref_y_list[i],
        #     mark_list[i][1],
        #     mark_list[i][2],
        #     time_step,
        #     np.array(
        #         [
        #             mark_list[i][0]
        #         ]
        #     ),
        #     np.array(
        #         [
        #             mark_list[i][0]
        #         ]
        #     ),
        #     [color_list_2[0]],
        #     [color_list_2[0]],
        #     FoV_letter_list[i],
        #     appendix=True
        # )


def make_fov_contour():

    f = h5py.File(old_kmeans_file, 'r')

    labels0 = f['new_final_labels'][0]

    f.close()

    shock_profiles = list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles)

    reverse_shock_profiles = list(weak_reverse_shocks_profiles) + list(strong_reverese_shock_profiles)

    quiet_profiles = list(set(range(100)) - set(shock_profiles) - set(reverse_shock_profiles))

    shock_mask = np.zeros((1236, 1848), dtype=np.int64)

    reverse_shock_mask = np.zeros((1236, 1848), dtype=np.int64)

    quiet_mask = np.zeros((1236, 1848), dtype=np.int64)

    for index, profile in enumerate(shock_profiles):
        shock_mask[np.where(labels0 == profile)] = index + 1

    for index, profile in enumerate(reverse_shock_profiles):
        reverse_shock_mask[np.where(labels0 == profile)] = index + 1

    for index, profile in enumerate(quiet_profiles):
        quiet_mask[np.where(labels0 == profile)] = 1

    shock_masked_array = np.ma.masked_array(shock_mask, shock_mask > 0)

    reverse_shock_masked_array = np.ma.masked_array(reverse_shock_mask, reverse_shock_mask > 0)

    quiet_masked_array = np.ma.masked_array(quiet_mask, quiet_mask > 0)

    extent = [596.31, 666.312, -35.041, 11.765]

    plt.close('all')

    plt.clf()

    plt.cla()

    fig , ax = plt.subplots(1, 1, figsize=(4.135, 5.845))

    ax.imshow(shock_masked_array, cmap='Blues', interpolation='nearest', origin='lower', extent=extent)

    ax.imshow(reverse_shock_masked_array, cmap='Reds', interpolation='nearest', origin='lower', extent=extent)

    ax.imshow(quiet_masked_array, cmap='gray', interpolation='nearest', origin='lower', extent=extent)

    ax.set_xlabel('x[arcsec]')

    ax.set_ylabel('y[arcsec]')

    fig.tight_layout()

    fig.savefig(
        'FOV_RP_map.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def get_response_function_data():
    inp_profiles = np.zeros((4, 29), dtype=np.float64)

    syn_profiles = np.zeros((4, 29), dtype=np.float64)

    response = np.zeros((4, 3, 150, 29), dtype=np.float64)

    # 0: vlos, 1: vturb, 2: temp

    atmos_param = np.zeros((4, 3, 150), dtype=np.float64)

    x = [662, 662 + 50]
    y = [708, 708 + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )
    f = h5py.File(out_file, 'r')

    inp_profiles[1] = f['all_profiles'][4, 25, 18, 0:29]
    inp_profiles[2] = f['all_profiles'][5, 25, 18, 0:29]
    inp_profiles[3] = f['all_profiles'][6, 25, 18, 0:29]

    f.close()

    f = h5py.File(input_87_17_14_file, 'r')

    inp_profiles[0] = f['profiles'][()][0, 0, 2, 4:33, 0]

    f.close()

    f = h5py.File(response_662_file, 'r')

    syn_profiles[1] = f['profiles'][()][0, 0, 0, 4:33, 0]
    syn_profiles[2] = f['profiles'][()][0, 0, 1, 4:33, 0]
    syn_profiles[3] = f['profiles'][()][0, 0, 2, 4:33, 0]

    f.close()

    f = h5py.File(response_87_17_14_file, 'r')

    syn_profiles[0] = f['profiles'][()][0, 0, 2, 4:33, 0]

    f.close()

    f = h5py.File(response_662_file, 'r')

    response[1, :] = f['derivatives'][()][0, 0, 0, np.array([1, 2, 0]), :, 4:33, 0]
    response[2, :] = f['derivatives'][()][0, 0, 1, np.array([1, 2, 0]), :, 4:33, 0]
    response[3, :] = f['derivatives'][()][0, 0, 2, np.array([1, 2, 0]), :, 4:33, 0]

    f.close()

    f = h5py.File(response_87_17_14_file, 'r')

    response[0, :] = f['derivatives'][()][0, 0, 2, np.array([1, 2, 0]), :, 4:33, 0]

    f.close()

    f = h5py.File(out_file, 'r')

    atmos_param[1, 0, :] = f['all_vlos'][4, 25, 18]
    atmos_param[2, 0, :] = f['all_vlos'][5, 25, 18]
    atmos_param[3, 0, :] = f['all_vlos'][6, 25, 18]

    atmos_param[1, 1, :] = f['all_vturb'][4, 25, 18]
    atmos_param[2, 1, :] = f['all_vturb'][5, 25, 18]
    atmos_param[3, 1, :] = f['all_vturb'][6, 25, 18]

    atmos_param[1, 2, :] = f['all_temp'][4, 25, 18]
    atmos_param[2, 2, :] = f['all_temp'][5, 25, 18]
    atmos_param[3, 2, :] = f['all_temp'][6, 25, 18]

    f.close()

    f = h5py.File(output_atmos_87_17_14_file, 'r')

    atmos_param[0, 0, :] = f['vlos'][0, 0, 2]
    atmos_param[0, 1, :] = f['vturb'][0, 0, 2]
    atmos_param[0, 2, :] = f['temp'][0, 0, 2]

    f.close()

    atmos_param[:, 2] /= 1e3

    atmos_param[0, 0] -= calib_velocity

    atmos_param[0, 0:2] /= 1e5

    for ii in range(4):
        for jj in range(3):
            response[ii, jj] /= np.abs(response[ii, jj]).max()

    return inp_profiles, syn_profiles, response, atmos_param


def plot_response_functions():
    write_path = Path('/home/harsh/Shocks Paper/')

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        2
    )

    size = plt.rcParams['lines.markersize']

    inp_profiles, syn_profiles, response, atmos_param = get_response_function_data()

    fontsize = 8

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(
        figsize=(
            6,
            6
        )
    )
    gs = gridspec.GridSpec(3, 3)

    gs.update(wspace=0.0, hspace=0.0)

    relative_wave = get_relative_velocity(wave_3933[:-1])

    X, Y = np.meshgrid(relative_wave, ltau)

    k = 0

    for i in range(3):
        for j in range(3):
            axs = plt.subplot(gs[k])

            if j == 2:
                vmin = 0
                vmax = 1
                cmap = 'Greys'
            else:
                vmin = -1
                vmax = 1
                cmap='RdGy'

            if i == 0:
                tind = 4
            elif i == 1:
                tind = 5
            else:
                tind = 6

            im = axs.pcolormesh(
                X,
                Y,
                response[i + 1, j], cmap=cmap,
                shading='nearest',
                vmin=vmin,
                vmax=vmax
            )

            axs.invert_yaxis()

            axs.set_xticklabels([])

            axs.set_yticklabels([])

            axs2 = axs.twiny()

            axs2.plot(atmos_param[i+1, j], ltau, color='red', linewidth=0.5)

            axs2.plot(atmos_param[0, j], ltau, color='gray', linewidth=0.5)

            axs2.set_xticks([])

            axs2.set_xticklabels([])

            axs2.set_yticklabels([])

            axs.set_yticks([0,  -1, -2, -3, -4, -5, -6, -7])

            if i == 2:
                axs.set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)

                axs.set_xticks([-0.5, 0, 0.5])
                axs.set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)

                cbaxes = inset_axes(
                    axs, width="30%", height="3%",
                    loc=3, borderpad=0,
                    bbox_to_anchor=[0.07, 0.05, 1, 1],
                    bbox_transform=axs.transAxes
                )
                cbar = fig.colorbar(
                    im,
                    cax=cbaxes,
                    ticks=[vmin, vmax],
                    orientation='horizontal'
                )
                cbar.ax.xaxis.set_ticks_position('top')

                cbar.ax.tick_params(labelsize=fontsize, colors='black')

            if j == 0:

                axs.set_ylabel(r'$\log \tau_{\mathrm{500}}$', fontsize=fontsize)
                axs.set_yticklabels([0, -1, -2, -3, -4, -5, -6, -7], fontsize=fontsize)

                axs.text(
                    0.65, 0.05, '{} s'.format(time[tind]),
                    transform=axs.transAxes,
                    color='black',
                    fontsize=fontsize
                )

                axs2.set_xlim(-8, 8)
                
                if i == 0:
                    axs.text(
                        0.05, 0.9, 'ROI A',
                        transform=axs.transAxes,
                        color='black',
                        fontsize=fontsize
                    )
                    axs2.set_xticks([-5, 0, 5])
                    axs2.set_xlabel(r'$V_{\mathrm{LOS}}\;\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
                    axs2.set_xticklabels([-5, 0, 5], fontsize=fontsize)

            if j == 1:
                
                axs2.set_xlim(0, 5)

                if i == 0:
                    axs2.set_xticks([0, 1, 2, 3, 4, 5, 6])
                    axs2.set_xlabel(r'$V_{\mathrm{turb}}\;\mathrm{[km\;s^{-1}]}$', fontsize=fontsize)
                    axs2.set_xticklabels([0, 1, 2, 3, 4, 5, 6], fontsize=fontsize)
                    
            if j == 2:

                axs2.set_xlim(3.5, 10)
                
                if i == 0:
                    axs2.set_xticks([4, 5, 6, 7, 8, 9, 10])
                    axs2.set_xlabel(r'$T\;\mathrm{[kK]}$', fontsize=fontsize)
                    axs2.set_xticklabels([4, 5, 6, 7, 8, 9, 10], fontsize=fontsize)

                axs3 = axs.twinx()

                axs3.scatter(relative_wave, inp_profiles[i+1, :], s=size/4, color='red')

                axs3.plot(relative_wave, syn_profiles[i+1, :], '--', linewidth=0.5, color='red')

                axs3.scatter(relative_wave, inp_profiles[0, :],  s=size/4, color='gray')

                axs3.plot(relative_wave, syn_profiles[0, :], '--', linewidth=0.5, color='gray')

                axs3.set_ylim(0, 0.4)

                axs3.set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
                axs3.set_yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                axs3.set_yticklabels([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], fontsize=fontsize)

                # max_indice_ltau, max_indice_wave = np.unravel_index(
                #     np.argmax(
                #         response[i+1, j, :, 10:17]
                #     ),
                #     response[i+1, j, :, 10:17].shape
                # )
                #
                # max_sf_tau = ltau[max_indice_ltau]
                #
                # max_sf_wave = relative_wave[max_indice_wave + 10]
                #
                # axs.plot(relative_wave, np.ones_like(relative_wave) * max_sf_tau, color='gray', linestyle='--', linewidth=0.5)
                # axs.axvline(x=max_sf_wave, color='gray', linestyle='--', linewidth=0.5)

            k += 1

    fig.savefig(
        write_path / 'Response_Functions.pdf',
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )

    fig.savefig(
        write_path / 'Response_Functions.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def get_data_for_intensity_enhancement_time_evolution(profile_array, classify_array):


    all_shock_profiles = list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles)

    data_t = list()

    data_intensity_enhancement = list()

    start = False
    start_indice = -1
    for i in range(100):
        if classify_array[i] in quiet_profiles:
            start = True
            start_indice = i
            continue
        if classify_array[i] in all_shock_profiles:

            minima_points = np.r_[True, profile_array[i][1:] < profile_array[i][:-1]] & np.r_[profile_array[i][:-1] < profile_array[i][1:], True]

            maxima_points = np.r_[False, profile_array[i][1:] > profile_array[i][:-1]] & np.r_[profile_array[i][:-1] > profile_array[i][1:], False]

            minima_indices = np.where(minima_points == True)[0]

            maxima_indices = np.where(maxima_points == True)[0]

            if maxima_indices.size == 1 and minima_indices.size == 2 and maxima_indices[0] <= 15:

                shock_intensity = (profile_array[i][maxima_indices[0]] - profile_array[i][minima_indices[0]]) / profile_array[i][minima_indices[0]]

                data_t.append(i - start_indice)

                data_intensity_enhancement.append(shock_intensity)

    return np.array(data_t), np.array(data_intensity_enhancement)


def get_kmeans_classification(ref_x, ref_y):

    f = h5py.File(old_kmeans_file, 'r')

    labels =  f['new_final_labels'][:, ref_x:ref_x+50, ref_y:ref_y+50]

    f.close()

    return labels


def get_all_profile_enhancement_data():
    data_t = list()

    data_intensity_enhancement = list()

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]
    f = h5py.File(old_kmeans_file, 'r')
    labels =  f['new_final_labels']

    total = 1236 * 1848

    last_progress = 0
    k = 0
    for i in range(1236):
        for j in range(1848):
            dt, de = get_data_for_intensity_enhancement_time_evolution(
                data[:, 0, 0:29, i, j], labels[:, i, j]
            )
            data_t += dt
            data_intensity_enhancement += de
            k += 1
            if k * 100 // total > last_progress:
                last_progress = k * 100 // total
                sys.stdout.write(
                    'Progress {}%'.format(
                        last_progress
                    )
                )

    return data_t, data_intensity_enhancement


if __name__ == '__main__':
    # make_shock_evolution_plots()
    plot_response_functions()
