import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
import time
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dateutil import parser
from helita.sim import rh15d
from scipy.interpolate import CubicSpline
from helita.sim import multi
from witt import *

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

cs00 = None
cs10 = None
cs20 = None
cs01 = None
cs11 = None
cs21 = None
cs02 = None
cs12 = None
cs22 = None
cs03 = None
cs13 = None
cs23 = None
cs04 = None
cs14 = None
cs24 = None


m1 = -1
s1 = 49

m2 = 1
s2 = 0

hx = np.arange(50)

hy1 = hx * m1 + s1
hy2 = hx * m2 + s2

atmos_indices0, atmos_indices1, atmos_indices2 = None, None, None

x = [662, 712]
y = [708, 758]

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


def get_supplementary_atmos_params(
    x, y, frames,
    output_atmos_quiet_filepath,
    output_atmos_shock_filepath,
    output_atmos_reverse_filepath,
    output_atmos_other_filepath,
    quiet_pixel_file,
    shock_pixel_file,
    reverse_pixel_file,
    other_pixel_file,
    output_atmos_shock_78_filepath=None,
    shock_78_pixel_file=None
):

    fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

    if output_atmos_shock_78_filepath:
        fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

    fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

    fout_atmos_other = h5py.File(
        output_atmos_other_filepath,
        'r'
    )

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

    nne = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    rho = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    pgas = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    z = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    nne[a1, b1, c1] = fout_atmos_quiet['nne'][0, 0]
    nne[a2, b2, c2] = fout_atmos_shock['nne'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        nne[a3, b3, c3] = fout_atmos_shock_78['nne'][0, 0]

    nne[a4, b4, c4] = fout_atmos_reverse['nne'][0, 0]
    nne[a6, b6, c6] = fout_atmos_other['nne'][0, 0]

    rho[a1, b1, c1] = fout_atmos_quiet['rho'][0, 0]
    rho[a2, b2, c2] = fout_atmos_shock['rho'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        rho[a3, b3, c3] = fout_atmos_shock_78['rho'][0, 0]

    rho[a4, b4, c4] = fout_atmos_reverse['rho'][0, 0]
    rho[a6, b6, c6] = fout_atmos_other['rho'][0, 0]

    pgas[a1, b1, c1] = fout_atmos_quiet['pgas'][0, 0]
    pgas[a2, b2, c2] = fout_atmos_shock['pgas'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        pgas[a3, b3, c3] = fout_atmos_shock_78['pgas'][0, 0]

    pgas[a4, b4, c4] = fout_atmos_reverse['pgas'][0, 0]
    pgas[a6, b6, c6] = fout_atmos_other['pgas'][0, 0]

    z[a1, b1, c1] = fout_atmos_quiet['z'][0, 0]
    z[a2, b2, c2] = fout_atmos_shock['z'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        z[a3, b3, c3] = fout_atmos_shock_78['z'][0, 0]

    z[a4, b4, c4] = fout_atmos_reverse['z'][0, 0]
    z[a6, b6, c6] = fout_atmos_other['z'][0, 0]

    return nne, pgas, rho, z


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
    global atmos_indices0, atmos_indices1, atmos_indices2

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

    atmos_indices0 = np.where(
        (fout_atmos_quiet['ltau500'][0, 0, 0] >= tau_indices_list[0][0]) &
        (fout_atmos_quiet['ltau500'][0, 0, 0] <= tau_indices_list[0][1])
    )[0]
    atmos_indices1 = np.where(
        (fout_atmos_quiet['ltau500'][0, 0, 0] >= tau_indices_list[1][0]) &
        (fout_atmos_quiet['ltau500'][0, 0, 0] <= tau_indices_list[1][1])
    )[0]
    atmos_indices2 = np.where(
        (fout_atmos_quiet['ltau500'][0, 0, 0] >= tau_indices_list[2][0]) &
        (fout_atmos_quiet['ltau500'][0, 0, 0] <= tau_indices_list[2][1])
    )[0]

    return all_profiles, syn_profiles, all_temp, all_vlos, all_vturb


def get_supplementary_output_filename_params(x, y, frameslist):
    return_tuple_list = list()

    base_path_input = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}'.format(
            x[0], x[1], y[0], y[1]
        )
    )
    base_path_inversions = base_path_input / 'plots'

    for frames, shock_78 in frameslist:

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
                (output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, other_pixel_file, output_atmos_shock_78_filepath, shock_78_pixel_file)
            )
        else:
            return_tuple_list.append(
                (output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, other_pixel_file)
            )

    return return_tuple_list


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


def get_supplementary_output_fov():

    global x, y

    frameslist = [([0, 21], True), ([21, 42], False), ([42, 63], True), ([63, 84], True), ([84, 100], True)]

    tuple_list = get_supplementary_output_filename_params(x, y, frameslist)

    nne, pgas, rho, z = None, None, None, None

    for index, a_tuple in enumerate(tuple_list):
        if nne is None:
            nne, pgas, rho, z = get_supplementary_atmos_params(
                x,
                y,
                frameslist[index][0],
                *a_tuple
            )
        else:
            a, b, c, d = get_supplementary_atmos_params(
                x,
                y,
                frameslist[index][0],
                *a_tuple
            )
            nne = np.vstack([nne, a])
            pgas = np.vstack([pgas, b])
            rho = np.vstack([rho, c])
            z = np.vstack([z, d])

    return nne, pgas, rho, z


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

    global cs00, cs01, cs02, cs03, cs04, cs10, cs11, cs12, cs13, cs14, cs20, cs21, cs22, cs23, cs24

    global atmos_indices0, atmos_indices1, atmos_indices2

    global x, y

    plt.cla()

    plt.clf()

    plt.close('all')

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    time_info, header_time = sunpy.io.fits.read(spectra_file_path)[5]

    all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = get_fov()

    X, Y = np.meshgrid(np.arange(50), np.arange(50))

    image00 = np.mean(
        all_profiles[0, :, :, wave_indices_list[0]],
        axis=0
    )

    image10 = np.mean(
        all_profiles[0, :, :, wave_indices_list[1]],
        axis=0
    )
    image20 = np.mean(
        all_profiles[0, :, :, wave_indices_list[2]],
        axis=0
    )

    synimage01 = np.mean(
        syn_profiles[0, :, :, wave_indices_list[0]],
        0
    )

    synimage11 = np.mean(
        syn_profiles[0, :, :, wave_indices_list[1]],
        0
    )

    synimage21 = np.mean(
        syn_profiles[0, :, :, wave_indices_list[2]],
        0
    )

    temp_map02 = np.mean(all_temp[0, :, :, atmos_indices0], axis=0)
    temp_map12 = np.mean(all_temp[0, :, :, atmos_indices1], axis=0)
    temp_map22 = np.mean(all_temp[0, :, :, atmos_indices2], axis=0)

    vlos_map03 = np.mean(
        all_vlos[0, :, :, atmos_indices0], axis=0
    )
    vlos_map13 = np.mean(
        all_vlos[0, :, :, atmos_indices1], axis=0
    )
    vlos_map23 = np.mean(
        all_vlos[0, :, :, atmos_indices2], axis=0
    )

    vturb_map04 = np.mean(all_vturb[0, :, :, atmos_indices0], axis=0)
    vturb_map14 = np.mean(all_vturb[0, :, :, atmos_indices1], axis=0)
    vturb_map24 = np.mean(all_vturb[0, :, :, atmos_indices2], axis=0)

    flabel = h5py.File(label_file, 'r')

    contour_mask = np.zeros((50, 50))

    sr, sc = list(), list()

    for profile in shock_proiles:
        asr, asc = np.where(flabel['new_final_labels'][0, x[0]:x[1], y[0]:y[1]] == profile)
        sr += list(asr)
        sc += list(asc)

    sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

    contour_mask[sr, sc] = 1

    fig, axs = plt.subplots(3, 5, figsize=(19.2, 10.8), dpi=100)

    im00 = axs[0][0].imshow(
        image00,
        origin='lower',
        cmap='gray'
    )
    im10 = axs[1][0].imshow(
        image10,
        origin='lower',
        cmap='gray'
    )
    im20 = axs[2][0].imshow(
        image20,
        origin='lower',
        cmap='gray'
    )
    im01 = axs[0][1].imshow(
        synimage01,
        origin='lower',
        cmap='gray'
    )
    im11 = axs[1][1].imshow(
        synimage11,
        origin='lower',
        cmap='gray'
    )
    im21 = axs[2][1].imshow(
        synimage21,
        origin='lower',
        cmap='gray'
    )

    im02 = axs[0][2].imshow(
        temp_map02,
        origin='lower',
        cmap='hot'
    )
    im12 = axs[1][2].imshow(
        temp_map12,
        origin='lower',
        cmap='hot'
    )
    im22 = axs[2][2].imshow(
        temp_map22,
        origin='lower',
        cmap='hot'
    )

    im03 = axs[0][3].imshow(
        vlos_map03,
        origin='lower',
        cmap='bwr',
        vmin=-10,
        vmax=10,
        aspect='equal',
    )
    im13 = axs[1][3].imshow(
        vlos_map13,
        origin='lower',
        cmap='bwr',
        vmin=-10,
        vmax=10,
        aspect='equal',
    )
    im23 = axs[2][3].imshow(
        vlos_map23,
        origin='lower',
        cmap='bwr',
        interpolation='none',
        vmin=-10,
        vmax=10,
        aspect='equal',
    )

    im04 = axs[0][4].imshow(
        vturb_map04,
        origin='lower',
        cmap='copper',
        vmin=0,
        vmax=5,
        aspect='equal',
    )
    im14 = axs[1][4].imshow(
        vturb_map14,
        origin='lower',
        cmap='copper',
        vmin=0,
        vmax=5,
        aspect='equal',
    )
    im24 = axs[2][4].imshow(
        vturb_map24,
        origin='lower',
        cmap='copper',
        vmin=0,
        vmax=5,
        aspect='equal',
    )

    cs00 = axs[0][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs01 = axs[1][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs02 = axs[2][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs03 = axs[0][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs04 = axs[1][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs10 = axs[2][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs11 = axs[0][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs12 = axs[1][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs13 = axs[2][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs14 = axs[0][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs20 = axs[1][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs21 = axs[2][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs22 = axs[0][4].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs23 = axs[1][4].contour(X, Y, contour_mask, levels=1, cmap='gray')
    cs24 = axs[2][4].contour(X, Y, contour_mask, levels=1, cmap='gray')

    x_tick_labels = [623.5, 623.87, 624.25, 624.63]

    y_tick_labels = [-9.59, -9.21, -8.83, -8.45]

    tick_position = [10, 20, 30, 40]

    axs[0][2].plot(hx, hy1)
    axs[1][2].plot(hx, hy1)
    axs[2][2].plot(hx, hy1)
    axs[0][3].plot(hx, hy1)
    axs[1][3].plot(hx, hy1)
    axs[2][3].plot(hx, hy1)
    axs[0][4].plot(hx, hy1)
    axs[1][4].plot(hx, hy1)
    axs[2][4].plot(hx, hy1)

    axs[0][2].plot(hx, hy2)
    axs[1][2].plot(hx, hy2)
    axs[2][2].plot(hx, hy2)
    axs[0][3].plot(hx, hy2)
    axs[1][3].plot(hx, hy2)
    axs[2][3].plot(hx, hy2)
    axs[0][4].plot(hx, hy2)
    axs[1][4].plot(hx, hy2)
    axs[2][4].plot(hx, hy2)

    axs[0][0].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[2][0].set_xticks(tick_position)
    axs[2][0].set_xticklabels(x_tick_labels, rotation=45)
    axs[0][1].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[2][1].set_xticks(tick_position)
    axs[2][1].set_xticklabels(x_tick_labels, rotation=45)
    axs[0][2].set_xticklabels([])
    axs[1][2].set_xticklabels([])
    axs[2][2].set_xticks(tick_position)
    axs[2][2].set_xticklabels(x_tick_labels, rotation=45)
    axs[0][3].set_xticklabels([])
    axs[1][3].set_xticklabels([])
    axs[2][3].set_xticks(tick_position)
    axs[2][3].set_xticklabels(x_tick_labels, rotation=45)
    axs[0][4].set_xticklabels([])
    axs[1][4].set_xticklabels([])
    axs[2][4].set_xticks(tick_position)
    axs[2][4].set_xticklabels(x_tick_labels, rotation=45)

    axs[0][0].set_yticks(tick_position)
    axs[0][0].set_yticklabels(y_tick_labels, rotation=45)
    axs[1][0].set_yticks(tick_position)
    axs[1][0].set_yticklabels(y_tick_labels, rotation=45)
    axs[2][0].set_yticks(tick_position)
    axs[2][0].set_yticklabels(y_tick_labels, rotation=45)
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[2][1].set_yticklabels([])
    axs[0][2].set_yticklabels([])
    axs[1][2].set_yticklabels([])
    axs[2][2].set_yticklabels([])
    axs[0][3].set_yticklabels([])
    axs[1][3].set_yticklabels([])
    axs[2][3].set_yticklabels([])
    axs[0][4].set_yticklabels([])
    axs[1][4].set_yticklabels([])
    axs[2][4].set_yticklabels([])

    cbar00 = fig.colorbar(im00, ax=axs[0][0])
    cbar10 = fig.colorbar(im10, ax=axs[1][0])
    cbar20 = fig.colorbar(im20, ax=axs[2][0])
    cbar01 = fig.colorbar(im01, ax=axs[0][1])
    cbar11 = fig.colorbar(im11, ax=axs[1][1])
    cbar21 = fig.colorbar(im21, ax=axs[2][1])
    cbar02 = fig.colorbar(im02, ax=axs[0][2])
    cbar12 = fig.colorbar(im12, ax=axs[1][2])
    cbar22 = fig.colorbar(im22, ax=axs[2][2])
    cbar03 = fig.colorbar(im03, ax=axs[0][3])
    cbar13 = fig.colorbar(im13, ax=axs[1][3])
    cbar23 = fig.colorbar(im23, ax=axs[2][3])
    cbar04 = fig.colorbar(im04, ax=axs[0][4])
    cbar14 = fig.colorbar(im14, ax=axs[1][4])
    cbar24 = fig.colorbar(im24, ax=axs[2][4])

    cbar00.ax.tick_params(labelsize=10)
    cbar10.ax.tick_params(labelsize=10)
    cbar20.ax.tick_params(labelsize=10)
    cbar01.ax.tick_params(labelsize=10)
    cbar11.ax.tick_params(labelsize=10)
    cbar21.ax.tick_params(labelsize=10)
    cbar02.ax.tick_params(labelsize=10)
    cbar12.ax.tick_params(labelsize=10)
    cbar22.ax.tick_params(labelsize=10)
    cbar03.ax.tick_params(labelsize=10)
    cbar13.ax.tick_params(labelsize=10)
    cbar23.ax.tick_params(labelsize=10)
    cbar04.ax.tick_params(labelsize=10)
    cbar14.ax.tick_params(labelsize=10)
    cbar24.ax.tick_params(labelsize=10)

    start_date = parser.parse(time_info[0][0][0, 0, 0, 0, 0])

    text = fig.text(0.5, 0.005, 't=0s', fontsize=12)

    def updatefig(j):

        global cs00, cs01, cs02, cs03, cs04, cs10, cs11, cs12, cs13, cs14, cs20, cs21, cs22, cs23, cs24

        contour_mask = np.zeros((50, 50))

        sr, sc = list(), list()

        for profile in shock_proiles:
            asr, asc = np.where(flabel['new_final_labels'][j, x[0]:x[1], y[0]:y[1]] == profile)
            sr += list(asr)
            sc += list(asc)

        sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

        contour_mask[sr, sc] = 1

        data00 = np.mean(
            all_profiles[j, :, :, wave_indices_list[0]],
            axis=0
        )

        data10 = np.mean(
            all_profiles[j, :, :, wave_indices_list[1]],
            axis=0
        )

        data20 = np.mean(
            all_profiles[j, :, :, wave_indices_list[2]],
            axis=0
        )

        data01 = np.mean(
            syn_profiles[j, :, :, wave_indices_list[0]],
            0
        )

        data11 = np.mean(
            syn_profiles[j, :, :, wave_indices_list[1]],
            0
        )

        data21 = np.mean(
            syn_profiles[j, :, :, wave_indices_list[2]],
            0
        )

        data02 = np.mean(
            all_temp[j, :, :, atmos_indices0],
            axis=0
        )

        data12 = np.mean(
            all_temp[j, :, :, atmos_indices1],
            axis=0
        )

        data22 = np.mean(
            all_temp[j, :, :, atmos_indices2],
            axis=0
        )

        data03 = np.mean(
            all_vlos[j, :, :, atmos_indices0], axis=0
        )

        data13 = np.mean(
            all_vlos[j, :, :, atmos_indices1], axis=0
        )

        data23 = np.mean(
            all_vlos[j, :, :, atmos_indices2], axis=0
        )

        data04 = np.mean(
            all_vturb[j, :, :, atmos_indices0], axis=0
        )

        data14 = np.mean(
            all_vturb[j, :, :, atmos_indices1], axis=0
        )

        data24 = np.mean(
            all_vturb[j, :, :, atmos_indices2], axis=0
        )

        im00.set_array(data00)
        im10.set_array(data10)
        im20.set_array(data20)
        im01.set_array(data01)
        im11.set_array(data11)
        im21.set_array(data21)
        im02.set_array(data02)
        im12.set_array(data12)
        im22.set_array(data22)
        im03.set_array(data03)
        im13.set_array(data13)
        im23.set_array(data23)
        im04.set_array(data04)
        im14.set_array(data14)
        im24.set_array(data24)


        im00.set_clim(
            data00.min(),
            data00.max()
        )
        im10.set_clim(
            data10.min(),
            data10.max()
        )
        im20.set_clim(
            data20.min(),
            data20.max()
        )
        im01.set_clim(
            data01.min(),
            data01.max()
        )
        im11.set_clim(
            data11.min(),
            data11.max()
        )
        im21.set_clim(
            data21.min(),
            data21.max()
        )
        im02.set_clim(
            data02.min(),
            data02.max()
        )
        im12.set_clim(
            data12.min(),
            data12.max()
        )
        im22.set_clim(
            data22.min(),
            data22.max()
        )
        # im03.set_clim(
        #     data03.min(),
        #     data03.max()
        # )
        # im13.set_clim(
        #     data13.min(),
        #     data13.max()
        # )
        # im23.set_clim(
        #     data23.min(),
        #     data23.max()
        # )
        # im04.set_clim(
        #     data04.min(),
        #     data04.max()
        # )
        # im14.set_clim(
        #     data14.min(),
        #     data14.max()
        # )
        # im24.set_clim(
        #     data24.min(),
        #     data24.max()
        # )

        for coll in cs00.collections:
            coll.remove()
        for coll in cs10.collections:
            coll.remove()
        for coll in cs20.collections:
            coll.remove()
        for coll in cs01.collections:
            coll.remove()
        for coll in cs11.collections:
            coll.remove()
        for coll in cs21.collections:
            coll.remove()
        for coll in cs02.collections:
            coll.remove()
        for coll in cs12.collections:
            coll.remove()
        for coll in cs22.collections:
            coll.remove()
        for coll in cs03.collections:
            coll.remove()
        for coll in cs13.collections:
            coll.remove()
        for coll in cs23.collections:
            coll.remove()
        for coll in cs04.collections:
            coll.remove()
        for coll in cs14.collections:
            coll.remove()
        for coll in cs24.collections:
            coll.remove()

        cs00 = axs[0][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs01 = axs[1][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs02 = axs[2][0].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs03 = axs[0][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs04 = axs[1][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs10 = axs[2][1].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs11 = axs[0][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs12 = axs[1][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs13 = axs[2][2].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs14 = axs[0][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs20 = axs[1][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs21 = axs[2][3].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs22 = axs[0][4].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs23 = axs[1][4].contour(X, Y, contour_mask, levels=1, cmap='gray')
        cs24 = axs[2][4].contour(X, Y, contour_mask, levels=1, cmap='gray')

        cur_date = parser.parse(time_info[0][0][j, 0, 0, 0, 0])

        time_diff = np.round((cur_date - start_date).total_seconds(), 2)

        text.set_text('t={}s'.format(time_diff))

        log(
            'Finished Frame {}'.format(
                j
            )
        )

        return [im00, im10, im20, im01, im11, im21, im02, im12, im22, im03, im13, im23, im04, im14, im24]

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


def combine_supplementary_outputs():
    nne, pgas, rho, z = get_supplementary_output_fov()

    f = h5py.File('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_662_712_708_758/plots/supplementary_outputs.h5', 'w')

    f['nne'] = nne

    f['pgas'] = pgas

    f['rho'] = rho

    f['z'] = z

    f.close()


def write_atmos_files(write_path, filename, list_of_points):
    base_path = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_662_712_708_758/plots/')

    fatmos = h5py.File(base_path / 'consolidated_results_velocity_calibrated_fov_662_712_708_758.h5', 'r')

    fsuppl = h5py.File(base_path / 'supplementary_outputs.h5', 'r')

    for index, point in enumerate(list_of_points):
        t, x, y = point

        cs = CubicSpline(ltau, fsuppl['z'][t, x, y])

        calib_z = cs(0)

        w = witt()

        pe_from_pg = np.vectorize(
            w.pe_from_pg
        )

        pe = pe_from_pg(
            fatmos['all_temp'][t, x, y],
            fsuppl['pgas'][t, x, y]
        )

        h6tpgpe = np.vectorize(
            w.getH6pop
        )

        h6pop = h6tpgpe(
            fatmos['all_temp'][t, x, y],
            fsuppl['pgas'][t, x, y],
            pe
        )

        multi.watmos_multi(
            str(write_path / filename) + '{}.atmos'.format(index),
            fatmos['all_temp'][t, x, y],
            fsuppl['nne'][t, x, y] / 1e6,
            z=(fsuppl['z'][t, x, y] - calib_z) / 1e5,
            vz=fatmos['all_vlos'][t, x, y] - 0.18,
            vturb=fatmos['all_vturb'][t, x, y],
            nh=h6pop,
            id='Bifrost {} {}'.format(x, y),
            scale='height'
        )

    fsuppl.close()

    fatmos.close()


def make_rh15d_file(write_path, filename, list_of_points):
    '''

    Args:
        write_path: pathlib.Path object
        filename: filename of the RH15d output file
        list_of_points: [[t1, x1, y1], [t2, x2, y2]...]

    Makes RH15D file at the write path
    Returns:
        None
    '''

    base_path = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_662_712_708_758/plots/')

    fatmos = h5py.File(base_path / 'consolidated_results_velocity_calibrated_fov_662_712_708_758.h5', 'r')

    fsuppl = h5py.File(base_path / 'supplementary_outputs.h5', 'r')

    T = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)
    vz = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)
    z = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)
    ne = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)
    rho = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)
    vturb = np.zeros((1, 1, len(list_of_points), 150), dtype=np.float64)

    for index, point in enumerate(list_of_points):
        t, x, y = point
        T[0, 0, index] = fatmos['all_temp'][t, x, y]
        vz[0, 0, index] = (fatmos['all_vlos'][t, x, y] - 0.18) * 1e3

        cs = CubicSpline(ltau, fsuppl['z'][t, x, y])

        calib_z = cs(0)

        z[0, 0, index] = (fsuppl['z'][t, x, y] - calib_z) / 1e2

        ne[0, 0, index] = fsuppl['nne'][t, x, y] * 1e6

        rho[0, 0, index] = fsuppl['rho'][t, x, y] * 1e6

        vturb[0, 0, index] = fatmos['all_vturb'][t, x, y] * 1e3

    rh15d.make_xarray_atmos(
        str(write_path / filename) + '.h5',
        T=T,
        vz=vz,
        z=z,
        ne=ne,
        rho=rho,
        vturb=vturb
    )

    fsuppl.close()

    fatmos.close()


if __name__ == '__main__':

    calib_velocity = None

    wave_indices_list = [
        photosphere_indices,
        mid_chromosphere_indices,
        upper_chromosphere_indices
    ]
    tau_indices_list = [
        photosphere_tau,
        mid_chromosphere_tau,
        upper_chromosphere_tau
    ]
    # plot_fov_parameter_variation(
    #     animation_path='inversion_map_fov_1.mp4'
    # )

    # combine_supplementary_outputs()

    write_path = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_662_712_708_758/plots/')

    filename = 'rh15datmos'

    list_of_points = [
        [4, 25, 18],
        [5, 25, 18],
        [6, 25, 18],
    ]
    write_atmos_files(write_path, filename, list_of_points)

    make_rh15d_file(write_path, filename, list_of_points)
