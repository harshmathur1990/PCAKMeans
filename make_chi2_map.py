import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

label_file = Path('/home/harsh/OsloAnalysis/new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5')

spectra_file_path = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')

input_profile_path = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/frame_0_21_x_662_712_y_708_758.nc')

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

y_grid = np.array(
    [
        -9.97172015, -9.93385081, -9.89598147, -9.85811214, -9.8202428 ,
        -9.78237346, -9.74450413, -9.70663479, -9.66876546, -9.63089612,
        -9.59302678, -9.55515745, -9.51728811, -9.47941877, -9.44154944,
        -9.4036801 , -9.36581076, -9.32794143, -9.29007209, -9.25220275,
        -9.21433342, -9.17646408, -9.13859474, -9.10072541, -9.06285607,
        -9.02498673, -8.9871174 , -8.94924806, -8.91137872, -8.87350939,
        -8.83564005, -8.79777071, -8.75990138, -8.72203204, -8.6841627 ,
        -8.64629337, -8.60842403, -8.57055469, -8.53268536, -8.49481602,
        -8.45694669, -8.41907735, -8.38120801, -8.34333868, -8.30546934,
        -8.2676    , -8.22973067, -8.19186133, -8.15399199, -8.11612266
    ]
)

x_grid = np.array(
    [
        623.12225839, 623.16012772, 623.19799706, 623.2358664 ,
        623.27373573, 623.31160507, 623.34947441, 623.38734374,
        623.42521308, 623.46308242, 623.50095175, 623.53882109,
        623.57669043, 623.61455976, 623.6524291 , 623.69029843,
        623.72816777, 623.76603711, 623.80390644, 623.84177578,
        623.87964512, 623.91751445, 623.95538379, 623.99325313,
        624.03112246, 624.0689918 , 624.10686114, 624.14473047,
        624.18259981, 624.22046915, 624.25833848, 624.29620782,
        624.33407716, 624.37194649, 624.40981583, 624.44768517,
        624.4855545 , 624.52342384, 624.56129318, 624.59916251,
        624.63703185, 624.67490119, 624.71277052, 624.75063986,
        624.7885092 , 624.82637853, 624.86424787, 624.9021172 ,
        624.93998654, 624.97785588
    ]
)

cs00 = None


def plot_fov_parameter_variation(
    animation_path,
    wave_indices_list,
    tau_indices_list,
    fps=1
):

    global cs00

    plt.cla()

    plt.clf()

    plt.close('all')

    finputprofiles = h5py.File(input_profile_path, 'r')

    ind = np.where(finputprofiles['profiles'][0, 0, 0, :, 0] != 0)[0]

    data = finputprofiles['profiles'][:, :, :, ind, 0]

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

    fout = h5py.File(output_profile_filepath, 'r')

    fout_quiet = h5py.File(output_profile_quiet_filepath, 'r')

    fout_reverse = h5py.File(output_profile_reverse_shock_filepath, 'r')

    fout_other = h5py.File(output_profile_other_emission_filepath, 'r')

    fout_failed_falc = h5py.File(output_profile_failed_inversion_falc_filepath, 'r')

    fout_failed_falc_2 = h5py.File(output_profile_failed_inversion_falc_2_filepath, 'r')

    all_profiles = fout['profiles'][:, :, :, ind, 0]

    all_profiles[a1, b1, c1] = fout_quiet['profiles'][0, 0, :, ind, 0]
    all_profiles[d1, e1, g1] = fout_reverse['profiles'][0, 0, :, ind, 0]
    all_profiles[h1, i1, j1] = fout_other['profiles'][0, 0, :, ind, 0]
    all_profiles[k1, l1, m1] = fout_failed_falc['profiles'][0, 0, :, ind, 0]
    all_profiles[n1, o1, p1] = fout_failed_falc_2['profiles'][0, 0, :, ind, 0]

    chi2_map = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    data, all_profiles
                )
            ),
            axis=3
        )
    )

    flabel = h5py.File(label_file, 'r')

    X, Y = np.meshgrid(np.arange(50), np.arange(50))
    contour_mask = np.zeros((50, 50))

    qr, qc = list(), list()

    for profile in quiet_profiles:
        aqr, aqc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
        qr += list(aqr)
        qc += list(aqc)

    qr, qc = np.array(qr, dtype=np.int64), np.array(qc, dtype=np.int64)

    sr, sc = list(), list()

    for profile in shock_proiles:
        asr, asc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
        sr += list(asr)
        sc += list(asc)

    sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

    rr, rc = list(), list()

    for profile in reverse_shock_profiles:
        arr, arc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
        rr += list(arr)
        rc += list(arc)

    rr, rc = np.array(rr, dtype=np.int64), np.array(rc, dtype=np.int64)

    otr, otc = list(), list()

    for profile in other_emission_profiles:
        aotr, aotc = np.where(flabel['new_final_labels'][0, 662:712, 708:758] == profile)
        otr += list(aotr)
        otc += list(aotc)

    otr, otc = np.array(otr, dtype=np.int64), np.array(otc, dtype=np.int64)

    contour_mask[qr, qc] = 0
    contour_mask[sr, sc] = 1
    contour_mask[rr, rc] = 2
    contour_mask[otr, otc] = 3

    im00 = plt.imshow(
        chi2_map[0],
        origin='lower',
        cmap='gray',
        interpolation='none',
        # extent=[x_grid[0], x_grid[1], y_grid[0], y_grid[1]]
    )

    cs00 = plt.contourf(X, Y, contour_mask, levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)

    x_tick_labels = [623.5, 623.87, 624.25, 624.63]

    y_tick_labels = [-9.59, -9.21, -8.83, -8.45]

    tick_position = [10, 20, 30, 40]

    plt.yticks(tick_position, y_tick_labels)
    plt.xticks(tick_position, x_tick_labels, rotation=45)

    cbar00 = plt.gcf().colorbar(im00, ax=plt.gca())

    cbar00.ax.tick_params(labelsize=10)

    artists, labels = cs00.legend_elements()

    labels = ['QS', 'Shock', 'Reverse Shock', 'Other Emission']

    plt.gcf().legend(artists, labels, loc='upper left')

    text = plt.gcf().text(0.01, 0.5, time_info[0][0][0, 0, 0, 0, 0], fontsize=14)

    def updatefig(j):

        global cs00

        contour_mask = np.zeros((50, 50))

        qr, qc = list(), list()

        for profile in quiet_profiles:
            aqr, aqc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
            qr += list(aqr)
            qc += list(aqc)

        qr, qc = np.array(qr, dtype=np.int64), np.array(qc, dtype=np.int64)

        sr, sc = list(), list()

        for profile in shock_proiles:
            asr, asc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
            sr += list(asr)
            sc += list(asc)

        sr, sc = np.array(sr, dtype=np.int64), np.array(sc, dtype=np.int64)

        rr, rc = list(), list()

        for profile in reverse_shock_profiles:
            arr, arc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
            rr += list(arr)
            rc += list(arc)

        rr, rc = np.array(rr, dtype=np.int64), np.array(rc, dtype=np.int64)

        otr, otc = list(), list()

        for profile in other_emission_profiles:
            aotr, aotc = np.where(flabel['new_final_labels'][j, 662:712, 708:758] == profile)
            otr += list(aotr)
            otc += list(aotc)

        otr, otc = np.array(otr, dtype=np.int64), np.array(otc, dtype=np.int64)

        contour_mask[qr, qc] = 0
        contour_mask[sr, sc] = 1
        contour_mask[rr, rc] = 2
        contour_mask[otr, otc] = 3
        # set the data in the axesimage object
        im00.set_array(
            chi2_map[j]
        )

        for coll in cs00.collections:
            coll.remove()

        cs00 = plt.contourf(X, Y, contour_mask, levels=[-0.5,0.5,1.5,2.5,3.5], hatches=['/', '.', '*', 'o'], cmap='gray', alpha=0.1)
        text.set_text(time_info[0][0][j, 0, 0, 0, 0])
        # return the artists set
        return [im00]

    rate = 1000 / fps

    plt.tight_layout()

    ani = animation.FuncAnimation(
        plt.gcf(),
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

    plt.close(plt.gcf())

    plt.close('all')


if __name__ == '__main__':

    calib_velocity = None

    plot_fov_parameter_variation(
        animation_path='chi2_map_fov_falc.mp4',
        wave_indices_list=[photosphere_indices, mid_chromosphere_indices,upper_chromosphere_indices],
        tau_indices_list=[photosphere_tau, mid_chromosphere_tau, upper_chromosphere_tau]
    )
