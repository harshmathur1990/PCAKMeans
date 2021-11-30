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

input_profile_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/inversions/frame_0_21_x_662_712_y_708_758.nc'
)

spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')

output_atmos_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1/frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_atmos.nc')

output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_quiet_v1/quiet_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_atmos.nc')

output_atmos_reverse_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_reverse_shock_v1/reverse_shock_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_atmos.nc')

output_atmos_other_emission_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_other_emission_v1/other_emission_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_atmos.nc')

output_atmos_failed_inversion_falc_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc/failed_inversions_falc_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_atmos.nc')

output_atmos_failed_inversion_falc_2_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc_2/failed_inversions_falc_2_frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_1_vt_4_atmos.nc')

output_profile_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1/frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_profs.nc')

output_profile_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_quiet_v1/quiet_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_profs.nc')

output_profile_reverse_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_reverse_shock_v1/reverse_shock_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_profs.nc')

output_profile_other_emission_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_other_emission_v1/other_emission_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_profs.nc')

output_profile_failed_inversion_falc_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc/failed_inversions_falc_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_profs.nc')

output_profile_failed_inversion_falc_2_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc_2/failed_inversions_falc_2_frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_1_vt_4_profs.nc')

reverse_shock_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_reverse_shock.h5')

quiet_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_new.h5')

other_emission_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_other_emission.h5')

failed_inversions_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_failed_inversions.h5')

failed_inversions_falc_2_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_failed_inversions_falc_2.h5')

quiet_profiles = [0, 11, 14, 15, 20, 21, 24, 28, 31, 34, 40, 42, 43, 47, 48, 51, 60, 62, 69, 70, 73, 74, 75, 86, 89, 90, 8, 44, 63, 84]

shock_proiles = [4, 10, 19, 26, 37, 52, 79, 85, 94, 1, 22, 23, 53, 55, 56, 66, 72, 77, 92, 99, 6, 49, 18, 36, 78]

reverse_shock_profiles = [3, 13, 16, 17, 25, 32, 33, 35, 41, 58, 61, 64, 68, 82, 95, 97]

other_emission_profiles = [2, 5, 7, 9, 12, 27, 29, 30, 38, 39, 45, 46, 50, 54, 57, 59, 65, 67, 71, 76, 80, 81, 83, 87, 88, 91, 93, 96, 98]

photosphere_indices = np.array([29])

mid_chromosphere_indices = np.array([4, 5, 6, 23, 24, 25])

upper_chromosphere_indices = np.arange(12, 18)

photosphere_tau = np.array([-1, 0])

mid_chromosphere_tau = np.array([-4, -3])

upper_chromosphere_tau = np.array([-5, -4])

x = [662, 712]

y = [708, 758]


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


def plot_fov_parameter_variation(
    animation_path,
    wave_indices_list,
    tau_indices_list,
    fps=1
):

    global cs00, cs01, cs02, cs03, cs04, cs10, cs11, cs12, cs13, cs14, cs20, cs21, cs22, cs23, cs24

    plt.cla()

    plt.clf()

    plt.close('all')

    finputprofiles = h5py.File(input_profile_path, 'r')

    ind = np.where(finputprofiles['profiles'][0, 0, 0, :, 0] != 0)[0]

    data, header = sunpy.io.fits.read(spectra_file_path)[0]

    time_info, header_time = sunpy.io.fits.read(spectra_file_path)[5]

    sys.stdout.write('Read Spectra File\n')

    fquiet = h5py.File(quiet_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    freverse = h5py.File(reverse_shock_pixel_file, 'r')

    d1, e1, g1 = freverse['pixel_indices'][0:3]

    fother = h5py.File(other_emission_pixel_file, 'r')

    h1, i1, j1 = fother['pixel_indices'][0:3]

    ffailed = h5py.File(failed_inversions_pixel_file, 'r')

    k1, l1, m1 = ffailed['pixel_indices'][0:3]

    ffailed_falc_2 = h5py.File(failed_inversions_falc_2_pixel_file, 'r')

    n1, o1, p1 = ffailed_falc_2['pixel_indices'][0:3]

    calib_velocity = 333390.00079943583

    fout = h5py.File(output_atmos_filepath, 'r')

    fout_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_reverse = h5py.File(output_atmos_reverse_shock_filepath, 'r')

    fout_other = h5py.File(output_atmos_other_emission_filepath, 'r')

    fout_failed_falc = h5py.File(
        output_atmos_failed_inversion_falc_filepath,
        'r'
    )

    fout_failed_falc_2 = h5py.File(
        output_atmos_failed_inversion_falc_2_filepath,
        'r'
    )

    fout_profile = h5py.File(output_profile_filepath, 'r')

    fout_quiet_profile = h5py.File(output_profile_quiet_filepath, 'r')

    fout_reverse_profile = h5py.File(
        output_profile_reverse_shock_filepath,
        'r'
    )

    fout_other_profile = h5py.File(output_profile_other_emission_filepath, 'r')

    fout_failed_falc_profile = h5py.File(
        output_profile_failed_inversion_falc_filepath,
        'r'
    )

    fout_failed_falc_2_profile = h5py.File(
        output_profile_failed_inversion_falc_2_filepath,
        'r'
    )

    all_profiles = fout_profile['profiles'][:, :, :, ind, 0]

    all_profiles[a1, b1, c1] = fout_quiet_profile['profiles'][0, 0, :, ind, 0]
    all_profiles[d1, e1, g1] = fout_reverse_profile['profiles'][0, 0, :, ind, 0]
    all_profiles[h1, i1, j1] = fout_other_profile['profiles'][0, 0, :, ind, 0]
    all_profiles[k1, l1, m1] = fout_failed_falc_profile['profiles'][0, 0, :, ind, 0]
    all_profiles[n1, o1, p1] = fout_failed_falc_2_profile['profiles'][0, 0, :, ind, 0]

    all_temp = fout['temp'][()]

    all_vlos = fout['vlos'][()]

    all_vturb = fout['vturb'][()]

    all_temp[a1, b1, c1] = fout_quiet['temp'][0, 0]
    all_temp[d1, e1, g1] = fout_reverse['temp'][0, 0]
    all_temp[h1, i1, j1] = fout_other['temp'][0, 0]
    all_temp[k1, l1, m1] = fout_failed_falc['temp'][0, 0]
    all_temp[n1, o1, p1] = fout_failed_falc_2['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_quiet['vlos'][0, 0]
    all_vlos[d1, e1, g1] = fout_reverse['vlos'][0, 0]
    all_vlos[h1, i1, j1] = fout_other['vlos'][0, 0]
    all_vlos[k1, l1, m1] = fout_failed_falc['vlos'][0, 0]
    all_vlos[n1, o1, p1] = fout_failed_falc_2['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_quiet['vturb'][0, 0]
    all_vturb[d1, e1, g1] = fout_reverse['vturb'][0, 0]
    all_vturb[h1, i1, j1] = fout_other['vturb'][0, 0]
    all_vturb[k1, l1, m1] = fout_failed_falc['vturb'][0, 0]
    all_vturb[n1, o1, p1] = fout_failed_falc_2['vturb'][0, 0]

    X, Y = np.meshgrid(np.arange(50), np.arange(50))

    atmos_indices0 = np.where(
        (fout['ltau500'][0, 0, 0] >= tau_indices_list[0][0]) &
        (fout['ltau500'][0, 0, 0] <= tau_indices_list[0][1])
    )[0]
    atmos_indices1 = np.where(
        (fout['ltau500'][0, 0, 0] >= tau_indices_list[1][0]) &
        (fout['ltau500'][0, 0, 0] <= tau_indices_list[1][1])
    )[0]
    atmos_indices2 = np.where(
        (fout['ltau500'][0, 0, 0] >= tau_indices_list[2][0]) &
        (fout['ltau500'][0, 0, 0] <= tau_indices_list[2][1])
    )[0]

    image00 = np.mean(
        finputprofiles['profiles'][0, :, :, ind[wave_indices_list[0]], 0],
        axis=2
    )

    image10 = np.mean(
        finputprofiles['profiles'][0, :, :, ind[wave_indices_list[1]], 0],
        axis=2
    )
    image20 = np.mean(
        finputprofiles['profiles'][0, :, :, ind[wave_indices_list[2]], 0],
        axis=2
    )

    synimage01 = np.mean(
        all_profiles[0, :, :, wave_indices_list[0]],
        0
    )

    synimage11 = np.mean(
        all_profiles[0, :, :, wave_indices_list[1]],
        0
    )

    synimage21 = np.mean(
        all_profiles[0, :, :, wave_indices_list[2]],
        0
    )

    temp_map02 = np.mean(all_temp[0, :, :, atmos_indices0], axis=0)
    temp_map12 = np.mean(all_temp[0, :, :, atmos_indices1], axis=0)
    temp_map22 = np.mean(all_temp[0, :, :, atmos_indices2], axis=0)

    vlos_map03 = np.mean(
        all_vlos[0, :, :, atmos_indices0] - calib_velocity, axis=0
    ) / 1e5
    vlos_map13 = np.mean(
        all_vlos[0, :, :, atmos_indices1] - calib_velocity, axis=0
    ) / 1e5
    vlos_map23 = np.mean(
        all_vlos[0, :, :, atmos_indices2] - calib_velocity, axis=0
    ) / 1e5

    vturb_map04 = np.mean(all_vturb[0, :, :, atmos_indices0], axis=0) / 1e5
    vturb_map14 = np.mean(all_vturb[0, :, :, atmos_indices1], axis=0) / 1e5
    vturb_map24 = np.mean(all_vturb[0, :, :, atmos_indices2], axis=0) / 1e5

    flabel = h5py.File(label_file, 'r')

    contour_mask = np.zeros((50, 50))

    # qr, qc = list(), list()

    # for profile in quiet_profiles:
    #     aqr, aqc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
    #     qr += list(aqr)
    #     qc += list(aqc)

    # qr, qc = np.array(qr, dtype=np.int64), np.array(qc, dtype=np.int64)

    sr, sc = list(), list()

    for profile in shock_proiles:
        asr, asc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
        sr += list(asr)
        sc += list(asc)

    sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

    # rr, rc = list(), list()

    # for profile in reverse_shock_profiles:
    #     arr, arc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
    #     rr += list(arr)
    #     rc += list(arc)

    # rr, rc = np.array(rr, dtype=np.int64), np.array(rc, dtype=np.int64)

    # otr, otc = list(), list()

    # for profile in other_emission_profiles:
    #     aotr, aotc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
    #     otr += list(aotr)
    #     otc += list(aotc)

    # otr, otc = np.array(otr, dtype=np.int64), np.array(otc, dtype=np.int64)

    # contour_mask[qr, qc] = 0
    contour_mask[sr, sc] = 1
    # contour_mask[rr, rc] = 2
    # contour_mask[otr, otc] = 3

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
        vmin=-6,
        vmax=6,
        aspect='equal',
    )
    im13 = axs[1][3].imshow(
        vlos_map13,
        origin='lower',
        cmap='bwr',
        vmin=-6,
        vmax=6,
        aspect='equal',
    )
    im23 = axs[2][3].imshow(
        vlos_map23,
        origin='lower',
        cmap='bwr',
        interpolation='none',
        vmin=-6,
        vmax=6,
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

    cs00 = axs[0][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs01 = axs[1][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs02 = axs[2][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs03 = axs[0][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs04 = axs[1][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs10 = axs[2][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs11 = axs[0][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs12 = axs[1][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs13 = axs[2][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs14 = axs[0][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs20 = axs[1][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs21 = axs[2][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs22 = axs[0][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs23 = axs[1][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
    cs24 = axs[2][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)

    x_tick_labels = [623.5, 623.87, 624.25, 624.63]

    y_tick_labels = [-9.59, -9.21, -8.83, -8.45]

    tick_position = [10, 20, 30, 40]

    axs[0][2].plot(np.ones(50) * 25)
    axs[1][2].plot(np.ones(50) * 25)
    axs[2][2].plot(np.ones(50) * 25)
    axs[0][3].plot(np.ones(50) * 25)
    axs[1][3].plot(np.ones(50) * 25)
    axs[2][3].plot(np.ones(50) * 25)
    axs[0][4].plot(np.ones(50) * 25)
    axs[1][4].plot(np.ones(50) * 25)
    axs[2][4].plot(np.ones(50) * 25)

    axs[0][2].axvline(x=25)
    axs[1][2].axvline(x=25)
    axs[2][2].axvline(x=25)
    axs[0][3].axvline(x=25)
    axs[1][3].axvline(x=25)
    axs[2][3].axvline(x=25)
    axs[0][4].axvline(x=25)
    axs[1][4].axvline(x=25)
    axs[2][4].axvline(x=25)

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

    # artists, labels = cs24.legend_elements()

    # labels = ['QS', 'Shock', 'Reverse Shock', 'Other Emission']

    # fig.legend(artists, labels, loc='upper left')

    start_date = parser.parse(time_info[0][0][0, 0, 0, 0, 0])

    text = fig.text(0.5, 0.005, 't=0s', fontsize=12)

    def updatefig(j):

        global cs00, cs01, cs02, cs03, cs04, cs10, cs11, cs12, cs13, cs14, cs20, cs21, cs22, cs23, cs24

        contour_mask = np.zeros((50, 50))

        # qr, qc = list(), list()

        # for profile in quiet_profiles:
        #     aqr, aqc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
        #     qr += list(aqr)
        #     qc += list(aqc)

        # qr, qc = np.array(qr, dtype=np.int64), np.array(qc, dtype=np.int64)

        sr, sc = list(), list()

        for profile in shock_proiles:
            asr, asc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
            sr += list(asr)
            sc += list(asc)

        sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

        # rr, rc = list(), list()

        # for profile in reverse_shock_profiles:
        #     arr, arc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
        #     rr += list(arr)
        #     rc += list(arc)

        # rr, rc = np.array(rr, dtype=np.int64), np.array(rc, dtype=np.int64)

        # otr, otc = list(), list()

        # for profile in other_emission_profiles:
        #     aotr, aotc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
        #     otr += list(aotr)
        #     otc += list(aotc)

        # otr, otc = np.array(otr, dtype=np.int64), np.array(otc, dtype=np.int64)

        # contour_mask[qr, qc] = 0
        contour_mask[sr, sc] = 1
        # contour_mask[rr, rc] = 2
        # contour_mask[otr, otc] = 3

        im00.set_array(
            np.mean(
                finputprofiles['profiles'][j, :, :, ind[wave_indices_list[0]], 0],
                axis=2
            )
        )

        im10.set_array(
            np.mean(
                finputprofiles['profiles'][j, :, :, ind[wave_indices_list[1]], 0],
                axis=2
            )
        )

        im20.set_array(
            np.mean(
                finputprofiles['profiles'][j, :, :, ind[wave_indices_list[2]], 0],
                axis=2
            )
        )

        im01.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[0]],
                0
            )
        )
        im11.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[1]],
                0
            )
        )
        im21.set_array(
            np.mean(
                all_profiles[j, :, :, wave_indices_list[2]],
                0
            )
        )

        im02.set_array(np.mean(all_temp[j, :, :, atmos_indices0], axis=0))
        im12.set_array(np.mean(all_temp[j, :, :, atmos_indices1], axis=0))
        im22.set_array(np.mean(all_temp[j, :, :, atmos_indices2], axis=0))

        im03.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices0] - calib_velocity, axis=0
            ) / 1e5
        )
        im13.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices1] - calib_velocity, axis=0
            ) / 1e5
        )
        im23.set_array(
            np.mean(
                all_vlos[j, :, :, atmos_indices2] - calib_velocity, axis=0
            ) / 1e5
        )

        im04.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices0], axis=0
            ) / 1e5
        )
        im14.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices1], axis=0
            ) / 1e5
        )
        im24.set_array(
            np.mean(
                all_vturb[j, :, :, atmos_indices2], axis=0
            ) / 1e5
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

        cs00 = axs[0][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs01 = axs[1][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs02 = axs[2][0].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs03 = axs[0][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs04 = axs[1][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs10 = axs[2][1].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs11 = axs[0][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs12 = axs[1][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs13 = axs[2][2].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs14 = axs[0][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs20 = axs[1][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs21 = axs[2][3].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs22 = axs[0][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs23 = axs[1][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        cs24 = axs[2][4].contour(X, Y, contour_mask, levels=1, cmap='gray')#levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)

        cur_date = parser.parse(time_info[0][0][j, 0, 0, 0, 0])

        time_diff = np.round((cur_date - start_date).total_seconds(), 2)

        text.set_text('t={}s'.format(time_diff))

        return [im00, im10, im20, im01, im11, im21, im02, im12, im22, im03, im13, im23, im04, im14, im24]

    rate = 1000 / fps

    fig.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(21),
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
        animation_path='inversion_map_fov_falc.mp4',
        wave_indices_list=[photosphere_indices, mid_chromosphere_indices,upper_chromosphere_indices],
        tau_indices_list=[photosphere_tau, mid_chromosphere_tau, upper_chromosphere_tau]
    )
