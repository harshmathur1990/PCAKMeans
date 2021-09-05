import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from prepare_data import *


base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'


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


def get_doppler_velocity(wavelength, center_wavelength):
    return (wavelength - center_wavelength) * 2.99792458e5 / center_wavelength


@np.vectorize
def get_doppler_velocity_3950(wavelength):
    return get_doppler_velocity(wavelength, 3933.682)


@np.vectorize
def get_relative_velocity(wavelength):
    return wavelength - 3933.682


def plot_fov_results_for_a_pixel(
    xs, ys, ref_x, ref_y, t, fovName
):

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        2
    )

    size = plt.rcParams['lines.markersize']

    rela_wave = get_relative_velocity(wave_3933[:-1])

    f = h5py.File(out_file, 'r')

    all_profiles = f['all_profiles']
    syn_profiles = f['syn_profiles']
    all_temp = f['all_temp']
    all_vlos = f['all_vlos']
    all_vturb = f['all_vturb']

    #------------------------------#

    plt.cla()

    plt.clf()

    plt.close('all')

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(2, 2)

    # gs.update(wspace=0.75, hspace=0.75)

    #------------------------------#
    axs = fig.add_subplot(gs[0])

    axs.scatter(
        rela_wave,
        all_profiles[t, ref_x, ref_y, 0:29],
        s=size/4,
        color='#0A1931'
    )

    axs.plot(
        rela_wave,
        syn_profiles[t, ref_x, ref_y, 0:29],
        '--',
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$\lambda\;(\AA)$')
    axs.set_ylabel(r'$I/I_{c 4000 \AA}$')

    axs.set_xticks([-0.5, 0, 0.5])
    axs.set_xticklabels([-0.5, 0, 0.5])

    axs.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    axs.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])

    axs.set_ylim(0, 0.4)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[1])

    axs.plot(
        ltau,
        all_temp[t, ref_x, ref_y] / 1e3,
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$T[kK]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([3, 5, 7, 9, 11])
    axs.set_yticklabels([3, 5, 7, 9, 11])

    axs.set_ylim(3, 11)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[2])

    axs.plot(
        ltau,
        all_vlos[t, ref_x, ref_y],
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{LOS}[kms^{-1}]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([-6, -3, 0, 3, 6])
    axs.set_yticklabels([-6, -3, 0, 3, 6])

    axs.set_ylim(-8, 8)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    axs = fig.add_subplot(gs[3])

    axs.plot(
        ltau,
        all_vturb[t, ref_x, ref_y],
        linewidth=0.5,
        color='#232323'
    )

    axs.set_xlabel(r'$log(\tau_{500})$')
    axs.set_ylabel(r'$V_{turb}[kms^{-1}]$')

    axs.set_xticks([-6, -5, -4, -3, -2, -1, 0])
    axs.set_xticklabels([-6, -5, -4, -3, -2, -1, 0])

    axs.set_yticks([0, 2, 4, 6])
    axs.set_yticklabels([0, 2, 4, 6])

    axs.set_ylim(0, 6)
    axs.set_xlim(-7, 0)

    axs.text(
        0.05, 0.8, r'FoV {}'.format(
            fovName
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    axs.text(
        0.6, 0.8, r't={}s'.format(
            time[t]
        ),
        transform=axs.transAxes,
        color='black',
        fontsize='small'
    )

    #------------------------------#

    fig.tight_layout()

    fig.savefig(
        'fov_{}_{}_{}_{}_t_{}_x_{}_y_{}.pdf'.format(
            x[0], x[1], y[0], y[1], t, ref_x, ref_y
        ),
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close('all')

    plt.cla()

    plt.clf()

    f.close()


def get_kmeans_classification(ref_x, ref_y, time_start):
    f = h5py.File(old_kmeans_file, 'r')

    labels =  f['new_final_labels'][:, ref_x:ref_x+50, ref_y:ref_y+50][time_start: time_start + 7]

    f.close()

    return labels


def get_params_for_fov_maps(f, time_start, wave_indice, tau_min, tau_max):

    ind = np.where((ltau >= tau_min) & (ltau <= tau_max))[0]

    all_params = np.zeros(
        (7, 5, 50, 50),
        dtype=np.float64
    )

    all_params[:, 0] = f['all_profiles'][time_start:time_start + 7][:, :, :, wave_indice]
    all_params[:, 1] = f['syn_profiles'][time_start:time_start + 7][:, :, :, wave_indice]
    all_params[:, 2] = np.mean(
        f['all_temp'][time_start:time_start + 7][:, :, :, ind],
        3
    ) / 1e3
    all_params[:, 3] = np.mean(
        f['all_vlos'][time_start:time_start + 7][:, :, :, ind],
        3
    )
    all_params[:, 4] = np.mean(
        f['all_vturb'][time_start:time_start + 7][:, :, :, ind],
        3
    )

    return all_params


def make_fov_maps(xs, ys, time_start, fovName, wave_indice, tau_min, tau_max, std_limit=10, vlos_min=None, vlos_max=None, wave_name=None, line_cut_x=None, line_cut_t=None):

    shock_proiles = list(strong_shocks_profiles)

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        2
    )

    f = h5py.File(out_file, 'r')

    all_params = get_params_for_fov_maps(f, time_start, wave_indice, tau_min, tau_max)

    all_vmin = np.zeros(5)
    all_vmax = np.zeros(5)

    all_vmin[0] = np.round(all_params[:, 0].min(), 2)
    all_vmin[1] = np.round(all_params[:, 1].min(), 2)
    all_vmin[2] = np.round(all_params[:, 2].min(), 2)
    all_vmin[3] = -8 if vlos_min is None else vlos_min
    all_vmin[4] = 0

    all_vmax[0] = np.round(all_params[:, 0].max(), 2)
    all_vmax[1] = np.round(all_params[:, 1].max(), 2)
    if std_limit > 0:
        all_vmax[2] = np.round(all_params[:, 2].mean() + 10 * all_params[:, 2].std(), 0)
    else:
        all_vmax[2] = np.round(all_params[:, 2].max(), 2)
    all_vmax[3] = 8 if vlos_max is None else vlos_max
    all_vmax[4] = 6

    f.close()

    labels = get_kmeans_classification(x[0], y[0], time_start)

    contour_mask = np.zeros_like(labels, dtype=np.int64)

    for profile in shock_proiles:
        contour_mask[np.where(labels == profile)] = 1

    X, Y = np.meshgrid(range(50), range(50))

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(
        figsize=(
            5, 7
        )
    )

    gs = gridspec.GridSpec(7, 5)

    gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    k = 0

    doppler_wave = np.round(
        get_doppler_velocity_3950(wave_3933),
        2
    )

    for i, t in enumerate(range(time_start, time_start + 7)):
        for j in range(5):
            vmin = all_vmin[j]
            vmax = all_vmax[j]

            if j == 0:
                cmap='gray'
            elif j == 1:
                cmap = 'gray'
            elif j == 2:
                cmap='hot'
            elif j == 3:
                cmap='bwr'
            else:
                cmap='copper'

            axs = fig.add_subplot(gs[k])

            im = axs.imshow(
                all_params[i, j],
                cmap=cmap,
                origin='lower',
                vmin=vmin,
                vmax=vmax
            )

            axs.contour(
                X, Y,
                contour_mask[i],
                levels=1,
                cmap='gray'
            )

            axs.spines["top"].set_color("white")
            axs.spines["left"].set_color("white")
            axs.spines["bottom"].set_color("white")
            axs.spines["right"].set_color("white")

            axs.spines["top"].set_linewidth(0.1)
            axs.spines["left"].set_linewidth(0.1)
            axs.spines["bottom"].set_linewidth(0.1)
            axs.spines["right"].set_linewidth(0.1)

            axs.set_xticklabels([])
            axs.set_yticklabels([])

            if line_cut_x is not None and line_cut_t is not None:
                if t in list(line_cut_t):
                    axs.plot(
                        range(50),
                        np.ones(50) * line_cut_x,
                        '--',
                        linewidth=0.5
                    )

            if i == 1 and j == 4:
                axs.text(
                    0.1, 0.6, 'FoV {}'.format(fovName),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='x-small'
                )

            if i == 0:
                if j == 0:
                    axs.text(
                        0.1, 0.8, r'${}\;Kms^{{-1}}$'.format(
                            doppler_wave[wave_indice]
                        ) if wave_name is None else wave_name,
                        transform=axs.transAxes,
                        color='white',
                        fontsize='x-small'
                    )
                elif j == 1:
                    axs.text(
                        0.1, 0.8, r'${}\;Kms^{{-1}}$'.format(
                            doppler_wave[wave_indice]
                        ) if wave_name is None else wave_name,
                        transform=axs.transAxes,
                        color='white',
                        fontsize='x-small'
                    )
                elif j == 2:
                    axs.text(
                        0.05, 0.8, r'${}<\log (\tau_{{500}})<{}$'.format(
                            tau_min, tau_max
                        ),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='xx-small'
                    )
                elif j == 3:
                    axs.text(
                        0.05, 0.8, r'${}<\log (\tau_{{500}})<{}$'.format(
                            tau_min, tau_max
                        ),
                        transform=axs.transAxes,
                        color='black',
                        fontsize='xx-small'
                    )
                elif j == 4:
                    axs.text(
                        0.05, 0.8, r'${}<\log (\tau_{{500}})<{}$'.format(
                            tau_min, tau_max
                        ),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='xx-small'
                    )
                cbaxes = inset_axes(
                    axs,
                    width="30%",
                    height="3%",
                    loc=3,
                    # borderpad=5
                )
                cbar = fig.colorbar(
                    im,
                    cax=cbaxes,
                    ticks=[vmin, vmax],
                    orientation='horizontal'
                )
                cbar.ax.xaxis.set_ticks_position('top')
                if j == 3:
                    cbar.ax.tick_params(labelsize=6, colors='black')
                else:
                    cbar.ax.tick_params(labelsize=6, colors='white')
            if j == 0:
                axs.text(
                    0.1, 0.6, r'${}\;s$'.format(time[t]),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='x-small'
                )

            k += 1

    fig.savefig(
        'results_fov_map_{}_{}_{}_{}_t_{}_{}_wave_indice_{}_tau_{}_{}.pdf'.format(
            x[0], x[1], y[0], y[1], time_start, time_start + 7, wave_indice, tau_min, tau_max
        ),
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def generate_files_for_response_function(xs, ys, time_start, ref_x, ref_y):

    calib_velocity = -94841.87483891034

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    f = h5py.File(out_file, 'r')

    m = sp.model(nx=7, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 0.3

    m.temp[0, 0, :] = f['all_temp'][time_start:time_start+7, ref_x, ref_y]

    m.vlos[0, 0, :] = (f['all_vlos'][time_start:time_start+7, ref_x, ref_y] * 1e5) + calib_velocity

    m.vturb[0, 0, :] = f['all_vturb'][time_start:time_start+7, ref_x, ref_y] * 1e5

    f.close()

    m.write(
        'wholedata_x_{}_{}_y_{}_{}_ref_x_{}_ref_y_{}_t_{}_{}_output_model.nc'.format(
            x[0], x[1], y[0], y[1], ref_x, ref_y, time_start, time_start+7
        )
    )


def generate_files_for_response_function_for_line_cuts(xs, ys, time, ref_x):

    calib_velocity = -94841.87483891034

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    f = h5py.File(out_file, 'r')

    m = sp.model(nx=50, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 0.3

    m.temp[0, 0, :] = f['all_temp'][time, ref_x]

    m.vlos[0, 0, :] = (f['all_vlos'][time, ref_x] * 1e5) + calib_velocity

    m.vturb[0, 0, :] = f['all_vturb'][time, ref_x] * 1e5

    f.close()

    m.write(
        'wholedata_x_{}_{}_y_{}_{}_ref_x_{}_t_{}_output_model.nc'.format(
            x[0], x[1], y[0], y[1], ref_x, time
        )
    )



def make_line_cut_plots(xs, ys, ref_x, time_array, fovName):

    shock_proiles = list(strong_shocks_profiles)

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        2
    )

    size = plt.rcParams['lines.markersize']

    labels_f = h5py.File(old_kmeans_file, 'r')

    labels = labels_f['new_final_labels'][time_array][:, xs + ref_x, ys:ys + 50]

    labels_mask = np.zeros((2, 150, 50), dtype=np.int64)

    for profile in shock_proiles:
        labels_mask[0][:, np.where(labels[0] == profile)] = 1
        labels_mask[1][:, np.where(labels[1] == profile)] = 1

    f = h5py.File(out_file, 'r')

    all_profiles = f['all_profiles']
    syn_profiles = f['syn_profiles']
    all_temp = f['all_temp']
    all_vlos = f['all_vlos']
    all_vturb = f['all_vturb']

    all_params = np.zeros((2, 3, 150, 50), dtype=np.float64)
    all_params[0, 0] = np.transpose(
        all_temp[time_array[0], ref_x],
        axes=(1, 0)
    ) / 1e3
    all_params[0, 1] = np.transpose(
        all_vlos[time_array[0], ref_x],
        axes=(1, 0)
    )
    all_params[0, 2] = np.transpose(
        all_vturb[time_array[0], ref_x],
        axes=(1, 0)
    )
    all_params[1, 0] = np.transpose(
        all_temp[time_array[1], ref_x],
        axes=(1, 0)
    ) / 1e3
    all_params[1, 1] = np.transpose(
        all_vlos[time_array[1], ref_x],
        axes=(1, 0)
    )
    all_params[1, 2] = np.transpose(
        all_vturb[time_array[1], ref_x],
        axes=(1, 0)
    )

    f.close()

    all_vmin = np.zeros(3)
    all_vmax = np.zeros(3)

    all_vmin[0] = np.round(
        all_params[:, 0].min(),
        2
    )
    all_vmax[0] = np.round(
        all_params[:, 0].max(),
        2
    )

    all_vmin[1] = -8
    all_vmax[1] = 8

    all_vmin[2] = 0
    all_vmax[2] = 6

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(4.135, 2.925))

    gs = gridspec.GridSpec(2, 3)

    gs.update(wspace=0.0, hspace=0.0)

    X, Y = np.meshgrid(range(50), ltau)

    k = 0

    for i in range(2):
        for j in range(3):
            axs = fig.add_subplot(gs[k])

            if j == 0:
                cmap = 'hot'

            elif j == 1:
                cmap='bwr'
            else:
                cmap='copper'

            im = axs.pcolormesh(
                X, Y,
                all_params[i, j],
                cmap=cmap,
                vmin=all_vmin[j],
                vmax=all_vmax[j],
                shading='gouraud'
            )

            axs.tick_params(labelsize=6, colors='black')

            axs.invert_yaxis()

            axs.contour(
                X, Y,
                labels_mask[i],
                levels=0,
                cmap='gray'
            )

            axs.set_xticks([])
            axs.set_yticks([])

            axs.set_xticklabels([])
            axs.set_yticklabels([])


            if i == 0:

                cbaxes = inset_axes(
                    axs,
                    width="80%",
                    height="3%",
                    loc='upper center',
                    borderpad=-1
                )
                cbar = fig.colorbar(
                    im,
                    cax=cbaxes,
                    ticks=[all_vmin[j], all_vmax[j]],
                    orientation='horizontal'
                )
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.tick_params(labelsize=6, colors='black')

            if j == 0:

                axs.text(
                    0.05, 0.05, r't={}s'.format(
                        time[time_array[i]]
                    ),
                    transform=axs.transAxes,
                    color='white',
                    fontsize='small'
                )

                axs.set_yticks([-7, -6, -5, -4, -3, -2, -1, 0])
                axs.set_yticklabels([-7, -6, -5, -4, -3, -2, -1, 0])

                if i == 0:
                    axs.text(
                        0.05, 0.8, r'FoV {}'.format(
                            fovName
                        ),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='small'
                    )
            k += 1

            if i == 1:
                axs.set_xticks([10, 20, 30, 40, 50])
                axs.set_xticklabels([10, 20, 30, 40, 50])

    fig.savefig(
        'line_cut_xs_{}_ys_{}_ref_x_{}_t_{}_{}.pdf'.format(
            xs, ys, ref_x, time_array[0], time_array[1]
        ),
        format='pdf',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def get_data_for_inversion_density_plots(type='shock'):

    labels = np.zeros(2 * 100 * 50 * 50, dtype=np.int64)
    temp = np.zeros((2 * 100 * 50 * 50, 150), dtype=np.float64)
    vlos = np.zeros((2 * 100 * 50 * 50, 150), dtype=np.float64)
    vturb = np.zeros((2 * 100 * 50 * 50, 150), dtype=np.float64)
    profiles = np.zeros((2 * 100 * 50 * 50, 29), dtype=np.float64)

    xs = 662

    ys = 708

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    f = h5py.File(old_kmeans_file, 'r')

    labels_f =  f['new_final_labels'][:, x[0]:x[1], y[0]:y[1]]

    f.close()

    labels[0:100 * 50 * 50] = labels_f.reshape(100 * 50 * 50)

    f = h5py.File(out_file, 'r')


    temp[0:100 * 50 * 50] = f['all_temp'][()].reshape(100 * 50 * 50, 150) / 1e3
    vlos[0:100 * 50 * 50] = f['all_vlos'][()].reshape(100 * 50 * 50, 150)
    vturb[0:100 * 50 * 50] = f['all_vturb'][()].reshape(100 * 50 * 50, 150)
    profiles[0:100 * 50 * 50] = f['all_profiles'][:, :, :, 0:29].reshape(100 * 50 * 50, 29)

    f.close()

    xs = 520

    ys = 715

    x = [xs, xs + 50]
    y = [ys, ys + 50]

    out_file = '/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/fov_{}_{}_{}_{}/plots/consolidated_results_velocity_calibrated_fov_{}_{}_{}_{}.h5'.format(
        x[0], x[1], y[0], y[1], x[0], x[1], y[0], y[1]
    )

    f = h5py.File(old_kmeans_file, 'r')

    labels_f =  f['new_final_labels'][:, x[0]:x[1], y[0]:y[1]]

    f.close()

    labels[100 * 50 * 50:] = labels_f.reshape(100 * 50 * 50)

    f = h5py.File(out_file, 'r')

    temp[100 * 50 * 50:] = f['all_temp'][()].reshape(100 * 50 * 50, 150) / 1e3
    vlos[100 * 50 * 50:] = f['all_vlos'][()].reshape(100 * 50 * 50, 150)
    vturb[100 * 50 * 50:] = f['all_vturb'][()].reshape(100 * 50 * 50, 150)
    profiles[100 * 50 * 50:] = f['all_profiles'][:, :, :, 0:29].reshape(100 * 50 * 50, 29)

    f.close()

    return profiles, temp, vlos, vturb, labels


def make_inversion_density_plots():

    rela_wave = get_relative_velocity(wave_3933[:-1])

    profiles, temp, vlos, vturb, labels = get_data_for_inversion_density_plots()

    plt.close('all')

    plt.clf()

    plt.cla()

    fig = plt.figure(figsize=(4, 4))

    k = 0

    gs = gridspec.GridSpec(2, 2)

    gs.update(wspace=0.0, hspace=0.0)

    ind_shock = list()

    for profile in list(strong_shocks_profiles):
        ind_shock = np.array(list(ind_shock) + list(np.where(labels == profile)[0]))

    ind_non_shock = np.array(list(set(range(2 * 100 * 50 * 50)) - set(ind_shock)))

    k = 0

    for i in range(2):
        for j in range(2):
            print ('{}-{}'.format(i, j))

            if k == 0:
                param = profiles
                min_t = 0
                max_t = 0.5
                x_bin = rela_wave
            elif k == 1:
                param = temp
                min_t = 3
                max_t = 12
                x_bin = ltau
            elif k == 2:
                param = vlos
                min_t = -8
                max_t = 8
                x_bin = ltau
            else:
                param = vturb
                min_t = 0
                max_t = 6
                x_bin = ltau

            axs = fig.add_subplot(gs[k]) #, label='1', frame_on=False)

            in_bins_t = np.linspace(min_t, max_t, 1000)

            center = np.mean(param[ind_non_shock], 0)

            H1, xedge1, yedge1 = np.histogram2d(
                np.tile(x_bin, ind_non_shock.shape[0]),
                param[ind_non_shock].flatten(),
                bins=(x_bin, in_bins_t)
            )

            X1, Y1 = np.meshgrid(xedge1, yedge1)

            axs.pcolormesh(X1, Y1, H1.T / np.abs(H1).max(), cmap='Reds')

            axs.plot(
                x_bin,
                center,
                color='red',
                linewidth=0.5,
                linestyle='solid'
            )

            center = np.mean(param[ind_shock], 0)

            H1, xedge1, yedge1 = np.histogram2d(
                np.tile(x_bin, ind_shock.shape[0]),
                param[ind_shock].flatten(),
                bins=(x_bin, in_bins_t)
            )

            X1, Y1 = np.meshgrid(xedge1, yedge1)

            axs.pcolormesh(X1, Y1, H1.T / np.abs(H1).max(), cmap='Blues')

            axs.plot(
                x_bin,
                center,
                color='blue',
                linewidth=0.5,
                linestyle='solid'
            )

            k += 1

    fig.savefig(
        'InversionDensityPlots.pdf',
        format='pdf',
        dpi=300
    )

    fig.savefig(
        'InversionDensityPlots.png',
        format='png',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


if __name__ == '__main__':

    make_inversion_density_plots()

