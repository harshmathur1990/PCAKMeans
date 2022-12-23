import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from skimage.exposure import adjust_gamma


base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

inversion_out_file = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/FoVAtoJ.nc')
inversion_out_file_alt = Path('/home/harsh/OsloAnalysis/new_kmeans/wholedata_inversions/FoVAtoJ_more_frames.h5')

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

@np.vectorize
def get_relative_velocity(wavelength):
    return wavelength - 3933.682


def make_inversion_fit_plot(xs, ys, wave_indice, time_steps, ref_x, ref_y, fovName):
    write_path = Path('/home/harsh/Shocks Paper/InversionMaps/')
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

    all_profiles = f['all_profiles'][time_steps][0:4]
    syn_profiles = f['syn_profiles'][time_steps][0:4]

    f.close()

    # vmin = [
    #     [
    #         all_profiles[:, :, :, wave_indice[0]].min(),
    #         syn_profiles[:, :, :, wave_indice[0]].min()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[1]].min(),
    #         syn_profiles[:, :, :, wave_indice[1]].min()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[2]].min(),
    #         syn_profiles[:, :, :, wave_indice[2]].min()
    #     ]
    # ]
    #
    # vmax = [
    #     [
    #         all_profiles[:, :, :, wave_indice[0]].max(),
    #         syn_profiles[:, :, :, wave_indice[0]].max()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[1]].max(),
    #         syn_profiles[:, :, :, wave_indice[1]].max()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[2]].max(),
    #         syn_profiles[:, :, :, wave_indice[2]].max()
    #     ]
    # ]

    vmin = np.minimum(all_profiles[:, :, :, wave_indice].min(), syn_profiles[:, :, :, wave_indice].min())

    vmax = np.maximum(all_profiles[:, :, :, wave_indice].max(), syn_profiles[:, :, :, wave_indice].max())

    color = ['blue', 'orange', 'brown', 'darkgreen']

    fontsize = 6

    for l in range(3):
        plt.cla()

        plt.clf()

        plt.close('all')

        fig = plt.figure(figsize=(1.4, 3.41))

        gs = gridspec.GridSpec(4, 2)

        gs.update(left=0, right=1, top=0.91, bottom=0.1, wspace=0.0, hspace=0.0)

        k = 0
        for i in range(4):
            for j in range(2):
                axs = fig.add_subplot(gs[k])

                if j == 0:
                    axs.imshow(
                        all_profiles[i, :, :, wave_indice[l]],
                        cmap='gray',
                        origin='lower',
                        vmin=vmin,  #[l][j],
                        vmax=vmax  #[l][j]
                    )
                    axs.scatter([ref_y], [ref_x], marker='+', color=color[i])
                    if i == 0:
                        axs.set_title(
                            r'Observed',
                            fontsize=fontsize
                        )
                    axs.text(
                        0.1, 0.7,
                        r't={}s'.format(time[time_steps[i]]),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='xx-small'
                    )
                else:
                    axs.imshow(
                        syn_profiles[i, :, :, wave_indice[l]],
                        cmap='gray',
                        origin='lower',
                        vmin=vmin,  #[l][j],
                        vmax=vmax  #[l][j]
                    )
                    if i == 0:
                        axs.set_title(
                            r'Synthesized',
                            fontsize=fontsize
                        )

                axs.set_xticks([])
                axs.set_xticklabels([])
                axs.set_yticks([])
                axs.set_yticklabels([])

                k += 1

        fig.suptitle(
            r'Ca II K ${0:+.1f}$ m$\mathrm{{\AA}}$'.format(
                np.round(
                    get_relative_velocity(
                        wave_3933[wave_indice[l]]
                    ) * 1000,
                    1
                )
            ),
            fontsize=fontsize
        )
        fig.savefig(write_path / 'CaIIk_fit_{}.pdf'.format(l), format='pdf', dpi=300)

    plt.cla()

    plt.clf()

    plt.close('all')

    fig, axs = plt.subplots(4, 3, figsize=(2.32, 3.41))
    plt.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.1, wspace=0.5, hspace=0.15)
    for i in range(4):
        for j in range(3):
            if j == 0:
                axs[i][j].scatter(wave_3933[:-1] - 3933.682, all_profiles[i, ref_x, ref_y, 0:29], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_3933[:-1] - 3933.682, syn_profiles[i, ref_x, ref_y, 0:29], linestyle='--', color=color[i], linewidth=0.5)
                axs[i][j].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
                if i == 0:
                    axs[i][j].set_title(r'Ca II K', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                if i == 3:
                    axs[i][j].set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
            elif j == 1:
                axs[i][j].scatter(wave_8542 - 8542.09, all_profiles[i, ref_x, ref_y, 30:30+20], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_8542 - 8542.09, syn_profiles[i, ref_x, ref_y, 30:30+20], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'Ca II 8542 $\mathrm{\AA}$', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-1, 0, 1])
                if i == 3:
                    axs[i][j].set_xticklabels([-1, 0, 1], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
                axs[i][j].set_xlim(-2.4, 2.4)
            else:
                axs[i][j].scatter(wave_6173 - 6173.3340, all_profiles[i, ref_x, ref_y, 30 + 20:30 + 20 + 14], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_6173 - 6173.3340, syn_profiles[i, ref_x, ref_y, 30 + 20:30 + 20 + 14], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'Fe I 6173 $\mathrm{\AA}$', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                if i == 3:
                    axs[i][j].set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
                axs[i][j].set_xlim(-0.7, 0.7)

            fmt = lambda x, y: np.round(x, 1)
            axs[i][j].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            axs[i][j].yaxis.set_tick_params(labelsize=fontsize)

    # fig.tight_layout()

    fig.savefig(write_path / 'Profile_fits.pdf', format='pdf', dpi=300)


def get_data_for_inversion_fit_plot_alternate(index, ref_x, ref_y, wave_indice, index_alt=None, frame_alt=None, frame_res=None):
    data = np.zeros((4, 6, 50, 50), dtype=np.float64)
    obs_profiles = np.zeros((4, 64), dtype=np.float64)
    syn_profiles = np.zeros((4, 64), dtype=np.float64)

    f = h5py.File(inversion_out_file, 'r')
    falt = None
    if index_alt is not None:
        falt = h5py.File(inversion_out_file_alt, 'r')

    observed_ind = np.array([0, 2, 4])

    synthesized_ind = np.array([1, 3, 5])

    if index_alt is None:
        data[:, observed_ind] = np.transpose(
            f['all_profiles'][index * 7: index * 7 + 4, :, :, wave_indice],
            axes=(0, 3, 1, 2)
        )
        data[:, synthesized_ind] = np.transpose(
            f['syn_profiles'][index * 7: index * 7 + 4, :, :, wave_indice],
            axes=(0, 3, 1, 2)
        )
        obs_profiles[:, :] = f['all_profiles'][index * 7: index * 7 + 4, ref_x, ref_y]
        syn_profiles[:, :] = f['syn_profiles'][index * 7: index * 7 + 4, ref_x, ref_y]
    else:
        for inddd, tindex in enumerate(frame_res):
            if tindex in frame_alt:
                indalt = index_alt[frame_alt.index(tindex)]
                data[inddd, observed_ind] = np.transpose(
                    falt['all_profiles'][indalt, :, :, wave_indice],
                    axes=(2, 0, 1)
                )
                data[inddd, synthesized_ind] = np.transpose(
                    falt['syn_profiles'][indalt, :, :, wave_indice],
                    axes=(2, 0, 1)
                )
                obs_profiles[inddd, :] = falt['all_profiles'][indalt, ref_x, ref_y]
                syn_profiles[inddd, :] = falt['syn_profiles'][indalt, ref_x, ref_y]
            else:
                indalt = index * 7 + inddd
                data[inddd, observed_ind] = np.transpose(
                    f['all_profiles'][indalt, :, :, wave_indice],
                    axes=(2, 0, 1)
                )
                data[inddd, synthesized_ind] = np.transpose(
                    f['syn_profiles'][indalt, :, :, wave_indice],
                    axes=(2, 0, 1)
                )
                obs_profiles[inddd, :] = f['all_profiles'][indalt, ref_x, ref_y]
                syn_profiles[inddd, :] = f['syn_profiles'][indalt, ref_x, ref_y]

    if falt is not None:
        falt.close()

    f.close()

    return data, obs_profiles, syn_profiles


def make_inversion_fit_plot_alternate(fovName, index, ref_x, ref_y, wave_indice, time_steps, index_alt=None, frame_alt=None, frame_res=None, gamma=1):

    data, obs_profiles, syn_profiles = get_data_for_inversion_fit_plot_alternate(
        index, ref_x, ref_y, wave_indice,
        index_alt=index_alt, frame_alt=frame_alt, frame_res=frame_res
    )

    write_path = Path('/home/harsh/Shocks Paper/InversionMaps/')

    time = np.round(
        np.arange(0, 8.26 * 100, 8.26),
        1
    )

    size = plt.rcParams['lines.markersize']

    rela_wave = get_relative_velocity(wave_3933[:-1])

    # vmin = [
    #     [
    #         all_profiles[:, :, :, wave_indice[0]].min(),
    #         syn_profiles[:, :, :, wave_indice[0]].min()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[1]].min(),
    #         syn_profiles[:, :, :, wave_indice[1]].min()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[2]].min(),
    #         syn_profiles[:, :, :, wave_indice[2]].min()
    #     ]
    # ]
    #
    # vmax = [
    #     [
    #         all_profiles[:, :, :, wave_indice[0]].max(),
    #         syn_profiles[:, :, :, wave_indice[0]].max()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[1]].max(),
    #         syn_profiles[:, :, :, wave_indice[1]].max()
    #     ],
    #     [
    #         all_profiles[:, :, :, wave_indice[2]].max(),
    #         syn_profiles[:, :, :, wave_indice[2]].max()
    #     ]
    # ]

    vmin = data.min()

    vmax = data.max()

    color = ['blue', 'orange', 'brown', 'darkgreen']

    fontsize = 6

    for l in range(3):
        plt.cla()

        plt.clf()

        plt.close('all')

        fig = plt.figure(figsize=(1.4, 3.41))

        gs = gridspec.GridSpec(4, 2)

        gs.update(left=0, right=1, top=0.91, bottom=0.1, wspace=0.0, hspace=0.0)

        k = 0
        for i in range(4):
            for j in range(2):
                axs = fig.add_subplot(gs[k])
                print ('{}-{}'.format(i, j + (l * 2)))
                if l == 2:
                    gamma = 1
                    vmin = data[:, 4:].min()
                    vmax = data[:, 4:].max()
                axs.imshow(
                    adjust_gamma(data[i, j + (l * 2)], gamma=gamma),
                    cmap='gray',
                    origin='lower',
                    vmin=vmin,  # [l][j],
                    vmax=vmax  # [l][j]
                )
                if j == 0:
                    axs.scatter([ref_y], [ref_x], marker='+', color=color[i])
                    if i == 0:
                        axs.set_title(
                            r'Observed',
                            fontsize=fontsize
                        )
                    axs.text(
                        0.1, 0.7,
                        r't={}s'.format(time[time_steps[i]]),
                        transform=axs.transAxes,
                        color='white',
                        fontsize='xx-small'
                    )
                else:
                    if i == 0:
                        axs.set_title(
                            r'Synthesized',
                            fontsize=fontsize
                        )

                axs.set_xticks([])
                axs.set_xticklabels([])
                axs.set_yticks([])
                axs.set_yticklabels([])

                k += 1

        if l in [0, 1]:
            fig.suptitle(
                r'Ca II K ${0:+.1f}$ m$\mathrm{{\AA}}$'.format(
                    np.round(
                        get_relative_velocity(
                            wave_3933[wave_indice[l]]
                        ) * 1000,
                        1
                    )
                ),
                fontsize=fontsize
            )
        else:
            fig.suptitle(
                r'Fe I ${0:+.1f}$ m$\mathrm{{\AA}}$'.format(
                    np.round(
                        (wave_6173[wave_indice[l] - 50] - 6173.3352) * 1000,
                        1
                    )
                ),
                fontsize=fontsize
            )
        fig.savefig(write_path / 'ROI_{}_CaIIk_fit_{}.pdf'.format(fovName, l), format='pdf', dpi=300)

    plt.cla()

    plt.clf()

    plt.close('all')

    fig, axs = plt.subplots(4, 3, figsize=(2.32, 3.41))
    plt.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.1, wspace=0.5, hspace=0.15)
    for i in range(4):
        for j in range(3):
            if j == 0:
                axs[i][j].scatter(wave_3933[:-1] - 3933.682, obs_profiles[i, 0:29], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_3933[:-1] - 3933.682, syn_profiles[i, 0:29], linestyle='--', color=color[i], linewidth=0.5)
                axs[i][j].set_ylabel(r'$I/I_{\mathrm{c}}$', fontsize=fontsize)
                if i == 0:
                    axs[i][j].set_title(r'Ca II K', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                if i == 3:
                    axs[i][j].set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
            elif j == 1:
                axs[i][j].scatter(wave_8542 - 8542.09, obs_profiles[i, 30:30+20], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_8542 - 8542.09, syn_profiles[i, 30:30+20], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'Ca II 8542 $\mathrm{\AA}$', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-1, 0, 1])
                if i == 3:
                    axs[i][j].set_xticklabels([-1, 0, 1], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
                axs[i][j].set_xlim(-2.4, 2.4)
            else:
                axs[i][j].scatter(wave_6173 - 6173.3340, obs_profiles[i, 30 + 20:30 + 20 + 14], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_6173 - 6173.3340, syn_profiles[i, 30 + 20:30 + 20 + 14], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'Fe I 6173 $\mathrm{\AA}$', fontsize=fontsize)
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda\;\mathrm{[\AA]}$', fontsize=fontsize)
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                if i == 3:
                    axs[i][j].set_xticklabels([-0.5, 0, 0.5], fontsize=fontsize)
                else:
                    axs[i][j].set_xticklabels([])
                axs[i][j].set_xlim(-0.7, 0.7)

            fmt = lambda x, y: np.round(x, 1)
            axs[i][j].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            axs[i][j].yaxis.set_tick_params(labelsize=fontsize)

    # fig.tight_layout()

    fig.savefig(write_path / 'ROI_{}_Profile_fits.pdf'.format(fovName), format='pdf', dpi=300)


if __name__ == '__main__':
    # make_inversion_fit_plot(662, 708, np.array([12, 14, 16]), np.array([4, 5, 6, 7]), 25, 18, 'A')
    make_inversion_fit_plot_alternate('A', 0, 25, 18, np.array([12, 14, 30 + 20]), np.array([3, 5, 6, 7]), index_alt=[0, 1, 2], frame_alt=[3, 11, 12], frame_res=[3, 5, 6, 7], gamma=0.7)
    make_inversion_fit_plot_alternate('B', 3, 29, 21, np.array([12, 14, 30 + 20]), np.array([30, 33, 34, 35]), index_alt=[3, 4, 5, 6], frame_alt=[30, 31, 39, 40], frame_res=[30, 33, 34, 35], gamma=0.7)