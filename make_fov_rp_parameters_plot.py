import sys
from pathlib import Path
import numpy as np
import h5py
import sunpy.io

base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

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


def make_fov_rps_plots():
    get_temp, get_vlos, get_vturb = get_old_rps_result_atmos()

    f = h5py.File(old_kmeans_file, 'r')

    labels = f['new_final_labels'][6].astype(np.int64)

    vec_get_temp = np.vectorize(get_temp)

    vec_get_vlos = np.vectorize(get_vlos)

    vec_get_vturb = np.vectorize(get_vturb)

    temp_map = get_temp(labels)

    vlos_map = get_vlos(labels)

    vturb_map = get_vturb(labels)

    #waste plot ignored
