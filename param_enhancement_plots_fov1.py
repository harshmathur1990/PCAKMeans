import sys
import time
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dateutil import parser


label_file = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
)

spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')

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

    global cs00, cs01, cs02, cs03, cs04, cs10, cs11, cs12, cs13, cs14, cs20, cs21, cs22, cs23, cs24

    global atmos_indices0, atmos_indices1, atmos_indices2

    global x, y

    plt.cla()

    plt.clf()

    plt.close('all')

    all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = get_fov()

    f = h5py.File(label_file, 'r')

    labels = f['new_final_labels'][:, x[0]:x[1], y[0]:y[1]]

    plt.cla()

    plt.close(fig)

    plt.close('all')


if __name__ == '__main__':

    calib_velocity = None

    
    plot_fov_parameter_variation(
        animation_path='inversion_map_fov_1.mp4'
    )
