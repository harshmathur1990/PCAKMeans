import sys
import time
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dateutil import parser


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

arcsec = np.array(
    [
        623.12  , 623.1577, 623.1954, 623.2331, 623.2708, 623.3085,
        623.3462, 623.3839, 623.4216, 623.4593, 623.497 , 623.5347,
        623.5724, 623.6101, 623.6478, 623.6855, 623.7232, 623.7609,
        623.7986, 623.8363, 623.874 , 623.9117, 623.9494, 623.9871,
        624.0248, 624.0625, 624.1002, 624.1379, 624.1756, 624.2133,
        624.251 , 624.2887, 624.3264, 624.3641, 624.4018, 624.4395,
        624.4772, 624.5149, 624.5526, 624.5903, 624.628 , 624.6657,
        624.7034, 624.7411, 624.7788, 624.8165, 624.8542, 624.8919,
        624.9296, 624.9673
    ]
)

label_file = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
)

spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 84, 8, 44, 63]

shock_proiles = [2, 4, 10, 19, 26, 30, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 67, 72, 77, 80, 81, 92, 87, 99, 36, 6, 49, 17, 96, 98, 78, 18]

reverse_shock_profiles = [3, 13, 16, 25, 32, 33, 35, 41, 45, 46, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [5, 7, 9, 12, 27, 29, 38, 39, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93]

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-5, -4])

cf00 = None
cf01 = None
cf02 = None
cf10 = None
cf11 = None
cf12 = None

x = [662, 712]

y = [708, 758]

m1 = -1
s1 = 49

m2 = 1
s2 = 0

hx = np.arange(50)

hy1 = hx * m1 + s1
hy2 = hx * m2 + s2

def log(logString):
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
    sys.stdout.write(
        '[{}] {}\n'.format(
            current_time,
            logString
        )
    )


def get_exact_filename(base_dir, starts_with, ends_with):
    all_files = base_dir.glob('**/*')

    for file in all_files:
        if file.name.startswith(starts_with) and file.name.endswith(ends_with):
            return file

    return None


def get_atmos_params(
    x, y, frames,
    input_profile_quiet,
    input_profile_shock,
    input_profile_reverse,
    input_profile_other,
    output_atmos_quiet_filepath,
    output_atmos_shock_filepath,
    output_atmos_reverse_filepath,
    output_atmos_other_filepath,
    output_profile_quiet_filepath,
    output_profile_shock_filepath,
    output_profile_reverse_filepath,
    output_profile_other_filepath,
    quiet_pixel_file,
    shock_pixel_file,
    reverse_pixel_file,
    other_pixel_file,
    input_profile_shock_78=None,
    output_atmos_shock_78_filepath=None,
    output_profile_shock_78_filepath=None,
    shock_78_pixel_file=None
):

    finputprofiles_quiet = h5py.File(input_profile_quiet, 'r')

    finputprofiles_shock = h5py.File(input_profile_shock, 'r')

    if input_profile_shock_78:
        finputprofiles_shock_78 = h5py.File(input_profile_shock_78, 'r')

    finputprofiles_reverse = h5py.File(input_profile_reverse, 'r')

    finputprofiles_other = h5py.File(input_profile_other, 'r')

    fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

    if output_atmos_shock_78_filepath:
        fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

    fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

    fout_atmos_other = h5py.File(
        output_atmos_other_filepath,
        'r'
    )

    fout_profile_quiet = h5py.File(output_profile_quiet_filepath, 'r')

    fout_profile_shock = h5py.File(output_profile_shock_filepath, 'r')

    if output_profile_shock_78_filepath:
        fout_profile_shock_78 = h5py.File(output_profile_shock_78_filepath, 'r')

    fout_profile_reverse = h5py.File(output_profile_reverse_filepath, 'r')

    fout_profile_other = h5py.File(output_profile_other_filepath, 'r')

    fquiet = h5py.File(quiet_pixel_file, 'r')

    fshock = h5py.File(shock_pixel_file, 'r')

    if shock_78_pixel_file:
        fshock_78 = h5py.File(shock_78_pixel_file, 'r')

    freverse = h5py.File(reverse_pixel_file, 'r')

    fother = h5py.File(other_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    a2, b2, c2 = fshock['pixel_indices'][0:3]

    if shock_78_pixel_file:
        a3, b3, c3 = fshock_78['pixel_indices'][0:3]

    a4, b4, c4 = freverse['pixel_indices'][0:3]

    a6, b6, c6 = fother['pixel_indices'][0:3]

    calib_velocity = -94841.87483891034

    all_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            64
        )
    )

    ind = np.where(finputprofiles_quiet['profiles'][0, 0, 0, :, 0] != 0)[0]

    all_profiles[a1, b1, c1] = finputprofiles_quiet['profiles'][0, 0, :, ind, 0]
    all_profiles[a2, b2, c2] = finputprofiles_shock['profiles'][0, 0, :, ind, 0]

    if shock_78_pixel_file and input_profile_shock_78:
        all_profiles[a3, b3, c3] = finputprofiles_shock_78['profiles'][0, 0, :, ind, 0]

    all_profiles[a4, b4, c4] = finputprofiles_reverse['profiles'][0, 0, :, ind, 0]
    all_profiles[a6, b6, c6] = finputprofiles_other['profiles'][0, 0, :, ind, 0]

    syn_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            64
        )
    )

    syn_profiles[a1, b1, c1] = fout_profile_quiet['profiles'][0, 0, :, ind, 0]
    syn_profiles[a2, b2, c2] = fout_profile_shock['profiles'][0, 0, :, ind, 0]

    if shock_78_pixel_file and output_profile_shock_78_filepath:
        syn_profiles[a3, b3, c3] = fout_profile_shock_78['profiles'][0, 0, :, ind, 0]

    syn_profiles[a4, b4, c4] = fout_profile_reverse['profiles'][0, 0, :, ind, 0]
    syn_profiles[a6, b6, c6] = fout_profile_other['profiles'][0, 0, :, ind, 0]

    all_temp = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vlos = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vturb = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_temp[a1, b1, c1] = fout_atmos_quiet['temp'][0, 0]
    all_temp[a2, b2, c2] = fout_atmos_shock['temp'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        all_temp[a3, b3, c3] = fout_atmos_shock_78['temp'][0, 0]

    all_temp[a4, b4, c4] = fout_atmos_reverse['temp'][0, 0]
    all_temp[a6, b6, c6] = fout_atmos_other['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_atmos_quiet['vlos'][0, 0]
    all_vlos[a2, b2, c2] = fout_atmos_shock['vlos'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        all_vlos[a3, b3, c3] = fout_atmos_shock_78['vlos'][0, 0]

    all_vlos[a4, b4, c4] = fout_atmos_reverse['vlos'][0, 0]
    all_vlos[a6, b6, c6] = fout_atmos_other['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_atmos_quiet['vturb'][0, 0]
    all_vturb[a2, b2, c2] = fout_atmos_shock['vturb'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        all_vturb[a3, b3, c3] = fout_atmos_shock_78['vturb'][0, 0]

    all_vturb[a4, b4, c4] = fout_atmos_reverse['vturb'][0, 0]
    all_vturb[a6, b6, c6] = fout_atmos_other['vturb'][0, 0]

    all_vlos = (all_vlos - calib_velocity) / 1e5

    all_vturb = all_vturb / 1e5

    return all_profiles, syn_profiles, all_temp, all_vlos, all_vturb


def get_filename_params(x, y, frameslist):
    return_tuple_list = list()

    base_path_input = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}'.format(
            x[0], x[1], y[0], y[1]
        )
    )
    base_path_inversions = base_path_input / 'plots'

    for frames, shock_78 in frameslist:
        input_profile_quiet = get_exact_filename(
            base_path_input,
            'wholedata_quiet_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.nc'
        )

        input_profile_shock = get_exact_filename(
            base_path_input,
            'wholedata_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.nc'
        )

        if shock_78:
            input_profile_shock_78 = get_exact_filename(
                base_path_input,
                'wholedata_shocks_78_18_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                    frames[0], frames[1], x[0], x[1], y[0], y[1]
                ),
                '.nc'
            )

        input_profile_reverse = get_exact_filename(
            base_path_input,
            'wholedata_reverse_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.nc'
        )

        input_profile_other = get_exact_filename(
            base_path_input,
            'wholedata_other_emission_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.nc'
        )

        output_atmos_quiet_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_quiet_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'atmos.nc'
        )

        output_atmos_shock_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'atmos.nc'
        )

        if shock_78:
            output_atmos_shock_78_filepath = get_exact_filename(
                base_path_inversions,
                'wholedata_shocks_78_18_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                    frames[0], frames[1], x[0], x[1], y[0], y[1]
                ),
                'atmos.nc'
            )

        output_atmos_reverse_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_reverse_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'atmos.nc'
        )

        output_atmos_other_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_other_emission_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'atmos.nc'
        )

        output_profile_quiet_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_quiet_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'profs.nc'
        )

        output_profile_shock_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'profs.nc'
        )

        if shock_78:
            output_profile_shock_78_filepath = get_exact_filename(
                base_path_inversions,
                'wholedata_shocks_78_18_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                    frames[0], frames[1], x[0], x[1], y[0], y[1]
                ),
                'profs.nc'
            )

        output_profile_reverse_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_reverse_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'profs.nc'
        )

        output_profile_other_filepath = get_exact_filename(
            base_path_inversions,
            'wholedata_other_emission_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            'profs.nc'
        )

        quiet_pixel_file = get_exact_filename(
            base_path_input,
            'pixel_indices_quiet_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.h5'
        )

        shock_pixel_file = get_exact_filename(
            base_path_input,
            'pixel_indices_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.h5'
        )

        if shock_78:
            shock_78_pixel_file = get_exact_filename(
                base_path_input,
                'pixel_indices_shocks_78_18_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                    frames[0], frames[1], x[0], x[1], y[0], y[1]
                ),
                '.h5'
            )

        reverse_pixel_file = get_exact_filename(
            base_path_input,
            'pixel_indices_reverse_shock_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.h5'
        )

        other_pixel_file = get_exact_filename(
            base_path_input,
            'pixel_indices_other_emission_profiles_rps_frame_{}_{}_x_{}_{}_y_{}_{}'.format(
                frames[0], frames[1], x[0], x[1], y[0], y[1]
            ),
            '.h5'
        )

        if shock_78:
            return_tuple_list.append(
                (input_profile_quiet, input_profile_shock, input_profile_reverse, input_profile_other, output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_other_filepath, output_profile_quiet_filepath, output_profile_shock_filepath, output_profile_reverse_filepath, output_profile_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, other_pixel_file, input_profile_shock_78, output_atmos_shock_78_filepath, output_profile_shock_78_filepath, shock_78_pixel_file)
            )
        else:
            return_tuple_list.append(
                (input_profile_quiet, input_profile_shock, input_profile_reverse, input_profile_other, output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_other_filepath, output_profile_quiet_filepath, output_profile_shock_filepath, output_profile_reverse_filepath, output_profile_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, other_pixel_file)
            )

    return return_tuple_list


def get_fov():

    global x, y

    frameslist = [([0, 21], True), ([21, 42], False), ([42, 63], True), ([63, 84], True), ([84, 100], True)]

    tuple_list = get_filename_params(x, y, frameslist)

    all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = None, None, None, None, None
    for index, a_tuple in enumerate(tuple_list):
        if all_profiles is None:
            all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = get_atmos_params(
                x,
                y,
                frameslist[index][0],
                *a_tuple
            )
        else:
            a, b, c, d, e = get_atmos_params(
                x,
                y,
                frameslist[index][0],
                *a_tuple
            )
            all_profiles = np.vstack([all_profiles, a])
            syn_profiles = np.vstack([syn_profiles, b])
            all_temp = np.vstack([all_temp, c])
            all_vlos = np.vstack([all_vlos, d])
            all_vturb = np.vstack([all_vturb, e])

    return all_profiles, syn_profiles, all_temp, all_vlos, all_vturb


def plot_fov_parameter_variation(
    animation_path,
    fps=1
):

    global cf00, cf01, cf02, cf10, cf11, cf12

    global x, y

    plt.cla()

    plt.clf()

    plt.close('all')

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    time_info, header_time = sunpy.io.fits.read(spectra_file_path)[5]

    all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = get_fov()

    fig, axs = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=100)

    cf00 = axs[0][0].pcolormesh(
        arcsec,
        ltau,
        all_temp[0, hx, hy1].T,
        shading='nearest',
        cmap='hot',
        vmin=4000,
        vmax=11000
    )
    cf01 = axs[0][1].pcolormesh(
        arcsec,
        ltau,
        all_vlos[0, hx, hy1].T,
        shading='nearest',
        cmap='bwr',
        vmin=-10,
        vmax=10
    )
    cf02 = axs[0][2].pcolormesh(
        arcsec,
        ltau,
        all_vturb[0, hx, hy1].T,
        shading='nearest',
        cmap='copper',
        vmin=0,
        vmax=5
    )

    cf10 = axs[1][0].pcolormesh(
        arcsec,
        ltau,
        all_temp[0, hx, hy2].T,
        shading='nearest',
        cmap='hot',
        vmin=4000,
        vmax=11000
    )
    cf11 = axs[1][1].pcolormesh(
        arcsec,
        ltau,
        all_vlos[0, hx, hy2].T,
        shading='nearest',
        cmap='bwr',
        vmin=-10,
        vmax=10
    )
    cf12 = axs[1][2].pcolormesh(
        arcsec,
        ltau,
        all_vturb[0, hx, hy2].T,
        shading='nearest',
        cmap='copper',
        vmin=0,
        vmax=5
    )
    axs[0][0].set_aspect(1.0 / axs[0][0].get_data_ratio(), adjustable='box')
    axs[0][1].set_aspect(1.0 / axs[0][1].get_data_ratio(), adjustable='box')
    axs[0][2].set_aspect(1.0 / axs[0][2].get_data_ratio(), adjustable='box')
    axs[1][0].set_aspect(1.0 / axs[1][0].get_data_ratio(), adjustable='box')
    axs[1][1].set_aspect(1.0 / axs[1][1].get_data_ratio(), adjustable='box')
    axs[1][2].set_aspect(1.0 / axs[1][2].get_data_ratio(), adjustable='box')

    x_tickposition = np.round([arcsec[10],arcsec[20], arcsec[30], arcsec[40]], 2)
    x_ticklabels = np.round([arcsec[10],arcsec[20], arcsec[30], arcsec[40]], 2)

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticklabels([])
    axs[1][0].set_xticks(x_tickposition)
    axs[1][0].set_xticklabels(x_ticklabels, rotation=15)
    axs[1][1].set_xticks(x_tickposition)
    axs[1][1].set_xticklabels(x_ticklabels, rotation=15)
    axs[1][2].set_xticks(x_tickposition)
    axs[1][2].set_xticklabels(x_ticklabels, rotation=15)

    axs[0][1].set_yticklabels([])
    axs[0][2].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])

    axs[0][0].invert_yaxis()
    axs[0][1].invert_yaxis()
    axs[0][2].invert_yaxis()
    axs[1][0].invert_yaxis()
    axs[1][1].invert_yaxis()
    axs[1][2].invert_yaxis()

    cbar00 = fig.colorbar(cf00, ax=axs[0][0])
    cbar01 = fig.colorbar(cf01, ax=axs[0][1])
    cbar02 = fig.colorbar(cf02, ax=axs[0][2])
    cbar10 = fig.colorbar(cf10, ax=axs[1][0])
    cbar11 = fig.colorbar(cf11, ax=axs[1][1])
    cbar12 = fig.colorbar(cf12, ax=axs[1][2])

    cbar00.ax.tick_params(labelsize=10)
    cbar01.ax.tick_params(labelsize=10)
    cbar02.ax.tick_params(labelsize=10)
    cbar10.ax.tick_params(labelsize=10)
    cbar11.ax.tick_params(labelsize=10)
    cbar12.ax.tick_params(labelsize=10)

    start_date = parser.parse(time_info[0][0][0, 0, 0, 0, 0])

    text = fig.text(0.5, 0.005, 't=0s', fontsize=10)

    def updatefig(j):
        global cf00, cf01, cf02, cf10, cf11, cf12

        cf00.set_array(
            all_temp[j, hx, hy1].T
        )
        cf01.set_array(
            all_vlos[j, hx, hy1].T
        )
        cf02.set_array(
            all_vturb[j, hx, hy1].T
        )

        cf10.set_array(
            all_temp[j, hx, hy2].T
        )
        cf11.set_array(
            all_vlos[j, hx, hy2].T
        )
        cf12.set_array(
            all_vturb[j, hx, hy2].T
        )

        cur_date = parser.parse(time_info[0][0][j, 0, 0, 0, 0])

        time_diff = np.round((cur_date - start_date).total_seconds(), 2)

        text.set_text('t={}s'.format(time_diff))

        return [cf00, cf01, cf02, cf10, cf11, cf12]

    rate = 1000 / fps

    fig.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(100),
        interval=rate,
        blit=True
    )

    Writer = animation.writers['ffmpeg']

    writer = Writer(
        fps=fps,
        metadata=dict(artist='Harsh Mathur'),
        bitrate=1800
    )

    ani.save(animation_path, writer=writer)

    plt.cla()

    plt.close(fig)

    plt.close('all')


if __name__ == '__main__':

    calib_velocity = None

    plot_fov_parameter_variation(
        animation_path='inversion_map_fov_1_line_cut.mp4'
    )
