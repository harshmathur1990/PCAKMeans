import sys
from pathlib import Path
import sunpy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
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

cf00 = None
cf01 = None
cf02 = None
cf10 = None
cf11 = None
cf12 = None


def plot_fov_parameter_variation(
    animation_path,
    fps=1
):

    global cf00, cf01, cf02, cf10, cf11, cf12

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

    all_vlos -= calib_velocity

    all_vlos /= 1e5

    all_vturb /= 1e5

    fig, axs = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=100)

    # import ipdb;ipdb.set_trace()
    cf00 = axs[0][0].pcolormesh(
        arcsec,
        ltau,
        all_temp[0, 25].T,
        shading='nearest',
        cmap='hot',
        vmin=4000,
        vmax=11000
    )
    cf01 = axs[0][1].pcolormesh(
        arcsec,
        ltau,
        all_vlos[0, 25].T,
        shading='nearest',
        cmap='bwr',
        vmin=-6,
        vmax=6
    )
    cf02 = axs[0][2].pcolormesh(
        arcsec,
        ltau,
        all_vturb[0, 25].T,
        shading='nearest',
        cmap='copper',
        vmin=0,
        vmax=5
    )

    cf10 = axs[1][0].pcolormesh(
        arcsec,
        ltau,
        all_temp[0, :, 25].T,
        shading='nearest',
        cmap='hot',
        vmin=4000,
        vmax=11000
    )
    cf11 = axs[1][1].pcolormesh(
        arcsec,
        ltau,
        all_vlos[0, :, 25].T,
        shading='nearest',
        cmap='bwr',
        vmin=-6,
        vmax=6
    )
    cf12 = axs[1][2].pcolormesh(
        arcsec,
        ltau,
        all_vturb[0, :, 25].T,
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

    # cbar00.ax.tick_params(labelsize=10)
    # cbar01.ax.tick_params(labelsize=10)
    # cbar02.ax.tick_params(labelsize=10)
    # cbar10.ax.tick_params(labelsize=10)
    # cbar11.ax.tick_params(labelsize=10)
    # cbar12.ax.tick_params(labelsize=10)

    start_date = parser.parse(time_info[0][0][0, 0, 0, 0, 0])

    text = fig.text(0.5, 0.005, 't=0s', fontsize=10)

    def updatefig(j):
        global cf00, cf01, cf02, cf10, cf11, cf12

        cf00.set_array(
            all_temp[j, 25].T
        )
        cf01.set_array(
            all_vlos[j, 25].T
        )
        cf02.set_array(
            all_vturb[j, 25].T
        )

        cf10.set_array(
            all_temp[j, :, 25].T
        )
        cf11.set_array(
            all_vlos[j, :, 25].T
        )
        cf12.set_array(
            all_vturb[j, :, 25].T
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
        animation_path='inversion_map_fov_1_line_cut.mp4'
    )