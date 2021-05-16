import sys
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

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 8, 44, 63, 84]

shock_proiles = [2, 4, 10, 17, 19, 26, 30, 37, 52, 67, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 72, 77, 80, 81, 92, 99, 6, 49, 18, 36, 78, 87, 96]

reverse_shock_profiles = [3, 13, 16, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [5, 7, 9, 12, 27, 29, 38, 39, 45, 46, 50, 54, 57, 59, 65, 71, 76, 83, 88, 91, 93, 98]

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

x = [770, 820]
y = [338, 388]


def get_atmos_params(
    x, y, frames,
    input_profile_quiet,
    input_profile_shock,
    input_profile_reverse,
    input_profile_retry,
    input_profile_other,
    output_atmos_quiet_filepath,
    output_atmos_shock_filepath,
    output_atmos_reverse_filepath,
    output_atmos_retry_filepath,
    output_atmos_other_filepath,
    output_profile_quiet_filepath,
    output_profile_shock_filepath,
    output_profile_reverse_filepath,
    output_profile_retry_filepath,
    output_profile_other_filepath,
    quiet_pixel_file,
    shock_pixel_file,
    reverse_pixel_file,
    retry_pixel_file,
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

    finputprofiles_retry = h5py.File(input_profile_retry, 'r')

    finputprofiles_other = h5py.File(input_profile_other, 'r')

    fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

    if output_atmos_shock_78_filepath:
        fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

    fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

    fout_atmos_retry = h5py.File(
        output_atmos_retry_filepath,
        'r'
    )

    fout_atmos_other = h5py.File(
        output_atmos_other_filepath,
        'r'
    )

    fout_profile_quiet = h5py.File(output_profile_quiet_filepath, 'r')

    fout_profile_shock = h5py.File(output_profile_shock_filepath, 'r')

    if output_profile_shock_78_filepath:
        fout_profile_shock_78 = h5py.File(output_profile_shock_78_filepath, 'r')

    fout_profile_reverse = h5py.File(output_profile_reverse_filepath, 'r')

    fout_profile_retry = h5py.File(output_profile_retry_filepath, 'r')

    fout_profile_other = h5py.File(output_profile_other_filepath, 'r')

    fquiet = h5py.File(quiet_pixel_file, 'r')

    fshock = h5py.File(shock_pixel_file, 'r')

    if shock_78_pixel_file:
        fshock_78 = h5py.File(shock_78_pixel_file, 'r')

    freverse = h5py.File(reverse_pixel_file, 'r')

    fretry = h5py.File(retry_pixel_file, 'r')

    fother = h5py.File(other_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    a2, b2, c2 = fshock['pixel_indices'][0:3]

    if shock_78_pixel_file:
        a3, b3, c3 = fshock_78['pixel_indices'][0:3]

    a4, b4, c4 = freverse['pixel_indices'][0:3]

    a5, b5, c5 = fretry['pixel_indices'][0:3]

    a6, b6, c6 = fother['pixel_indices'][0:3]

    calib_velocity = 333390.00079943583

    all_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            30
        )
    )

    ind = np.where(finputprofiles_quiet['profiles'][0, 0, 0, :, 0] != 0)[0]

    all_profiles[a1, b1, c1] = finputprofiles_quiet['profiles'][0, 0, :, ind, 0]
    all_profiles[a2, b2, c2] = finputprofiles_shock['profiles'][0, 0, :, ind, 0]

    if shock_78_pixel_file and input_profile_shock_78:
        all_profiles[a3, b3, c3] = finputprofiles_shock_78['profiles'][0, 0, :, ind, 0]

    all_profiles[a4, b4, c4] = finputprofiles_reverse['profiles'][0, 0, :, ind, 0]
    all_profiles[a5, b5, c5] = finputprofiles_retry['profiles'][0, 0, :, ind, 0]
    all_profiles[a6, b6, c6] = finputprofiles_other['profiles'][0, 0, :, ind, 0]

    syn_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            30
        )
    )

    syn_profiles[a1, b1, c1] = fout_profile_quiet['profiles'][0, 0, :, ind, 0]
    syn_profiles[a2, b2, c2] = fout_profile_shock['profiles'][0, 0, :, ind, 0]

    if shock_78_pixel_file and output_profile_shock_78_filepath:
        syn_profiles[a3, b3, c3] = fout_profile_shock_78['profiles'][0, 0, :, ind, 0]

    syn_profiles[a4, b4, c4] = fout_profile_reverse['profiles'][0, 0, :, ind, 0]
    syn_profiles[a5, b5, c5] = fout_profile_retry['profiles'][0, 0, :, ind, 0]
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
    all_temp[a5, b5, c5] = fout_atmos_retry['temp'][0, 0]
    all_temp[a6, b6, c6] = fout_atmos_other['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_atmos_quiet['vlos'][0, 0]
    all_vlos[a2, b2, c2] = fout_atmos_shock['vlos'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        all_vlos[a3, b3, c3] = fout_atmos_shock_78['vlos'][0, 0]

    all_vlos[a4, b4, c4] = fout_atmos_reverse['vlos'][0, 0]
    all_vlos[a5, b5, c5] = fout_atmos_retry['vlos'][0, 0]
    all_vlos[a6, b6, c6] = fout_atmos_other['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_atmos_quiet['vturb'][0, 0]
    all_vturb[a2, b2, c2] = fout_atmos_shock['vturb'][0, 0]

    if shock_78_pixel_file and output_atmos_shock_78_filepath:
        all_vturb[a3, b3, c3] = fout_atmos_shock_78['vturb'][0, 0]

    all_vturb[a4, b4, c4] = fout_atmos_reverse['vturb'][0, 0]
    all_vturb[a5, b5, c5] = fout_atmos_retry['vturb'][0, 0]
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


def get_filename_params(foldername, x, y, frameslist):
    return_tuple_list = list()

    for frames, shock_78 in frameslist:
        input_profile_quiet = Path(
            '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/quiet_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        input_profile_shock = Path(
            '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        if shock_78:
            input_profile_shock_78 = Path(
                '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_78_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                    foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
                )
            )

        input_profile_reverse = Path(
            '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/reverse_shock_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        input_profile_retry = Path(
            '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/retry_shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        input_profile_other = Path(
            '/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/other_emission_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/quiet_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_1_vt_4_atmos.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_atmos_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_atmos.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        if shock_78:
            output_atmos_shock_78_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_78_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_atmos.nc'.format(
                    foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
                )
            )

        output_atmos_reverse_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/reverse_shock_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_5_vt_4_atmos.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_atmos_retry_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/retry_shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_atmos.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_atmos_other_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/other_emission_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_5_vt_4_atmos.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_profile_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/quiet_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_1_vt_4_profs.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_profile_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_profs.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        if shock_78:
            output_profile_shock_78_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/shock_78_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_profs.nc'.format(
                    foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
                )
            )

        output_profile_reverse_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/reverse_shock_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_5_vt_4_profs.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_profile_retry_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/retry_shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_4_vl_5_vt_4_profs.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        output_profile_other_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/other_emission_profiles_frame_{}_{}_x_{}_{}_y_{}_{}_cycle_1_t_5_vl_5_vt_4_profs.nc'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        quiet_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_quiet_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        shock_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        if shock_78:
            shock_78_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_shock_78_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                    foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
                )
            )

        reverse_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_reverse_shock_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        retry_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_retry_shock_spicule_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        other_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/{}/pixel_indices_other_emission_profiles_frame_{}_{}_x_{}_{}_y_{}_{}.h5'.format(
                foldername, frames[0], frames[1], x[0], x[1], y[0], y[1]
            )
        )

        if shock_78:
            return_tuple_list.append(
                (input_profile_quiet, input_profile_shock, input_profile_reverse, input_profile_retry, input_profile_other, output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_retry_filepath, output_atmos_other_filepath, output_profile_quiet_filepath, output_profile_shock_filepath, output_profile_reverse_filepath, output_profile_retry_filepath, output_profile_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, retry_pixel_file, other_pixel_file, input_profile_shock_78, output_atmos_shock_78_filepath, output_profile_shock_78_filepath, shock_78_pixel_file)
            )
        else:
            return_tuple_list.append(
                (input_profile_quiet, input_profile_shock, input_profile_reverse, input_profile_retry, input_profile_other, output_atmos_quiet_filepath, output_atmos_shock_filepath, output_atmos_reverse_filepath, output_atmos_retry_filepath, output_atmos_other_filepath, output_profile_quiet_filepath, output_profile_shock_filepath, output_profile_reverse_filepath, output_profile_retry_filepath, output_profile_other_filepath, quiet_pixel_file, shock_pixel_file, reverse_pixel_file, retry_pixel_file, other_pixel_file)
            )

    return return_tuple_list


def get_fov():

    global x, y

    foldername = 'plots_v1_fifth_fov'
    frameslist = [([0, 21], False), ([21, 42], False), ([42, 56], False), ([56, 77], True), ([77, 100], False)]

    tuple_list = get_filename_params(foldername, x, y, frameslist)

    all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = None, None, None, None, None
    for index, a_tuple in enumerate(tuple_list):
        if all_profiles is None:
            all_profiles, syn_profiles, all_temp, all_vlos, all_vturb = get_atmos_params(
                [770, 820],
                [338, 388],
                frameslist[index][0],
                *a_tuple
            )
        else:
            a, b, c, d, e = get_atmos_params(
                [770, 820],
                [338, 388],
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

        im00.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[0]],
                axis=0
            )
        )

        im10.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[1]],
                axis=0
            )
        )

        im20.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[2]],
                axis=0
            )
        )

        im01.set_array(
            np.mean(
                syn_profiles[j, :, :, wave_indices_list[0]],
                0
            )
        )
        im11.set_array(
            np.mean(
                syn_profiles[j, :, :, wave_indices_list[1]],
                0
            )
        )
        im21.set_array(
            np.mean(
                syn_profiles[j, :, :, wave_indices_list[2]],
                0
            )
        )

        im02.set_array(np.mean(all_temp[j, :, :, atmos_indices0], axis=0))
        im12.set_array(np.mean(all_temp[j, :, :, atmos_indices1], axis=0))
        im22.set_array(np.mean(all_temp[j, :, :, atmos_indices2], axis=0))

        im03.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices0], axis=0
            )
        )
        im13.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices1], axis=0
            )
        )
        im23.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices2], axis=0
            )
        )

        im04.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices0], axis=0
            )
        )
        im14.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices1], axis=0
            )
        )
        im24.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices2], axis=0
            )
        )

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
    plot_fov_parameter_variation(
        animation_path='inversion_map_fov_2.mp4'
    )
