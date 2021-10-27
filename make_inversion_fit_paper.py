import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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

    vmin = [
        [
            all_profiles[:, :, :, wave_indice[0]].min(),
            syn_profiles[:, :, :, wave_indice[0]].min()
        ],
        [
            all_profiles[:, :, :, wave_indice[1]].min(),
            syn_profiles[:, :, :, wave_indice[1]].min()
        ],
        [
            all_profiles[:, :, :, wave_indice[2]].min(),
            syn_profiles[:, :, :, wave_indice[2]].min()
        ]
    ]

    vmax = [
        [
            all_profiles[:, :, :, wave_indice[0]].max(),
            syn_profiles[:, :, :, wave_indice[0]].max()
        ],
        [
            all_profiles[:, :, :, wave_indice[1]].max(),
            syn_profiles[:, :, :, wave_indice[1]].max()
        ],
        [
            all_profiles[:, :, :, wave_indice[2]].max(),
            syn_profiles[:, :, :, wave_indice[2]].max()
        ]
    ]

    color = ['blue', 'orange', 'brown', 'darkgreen']

    for l in range(3):
        plt.cla()

        plt.clf()

        plt.close('all')

        fig = plt.figure(figsize=(1.16, 2.56))

        gs = gridspec.GridSpec(4, 2)

        gs.update(left=0, right=1, top=0.9, bottom=0, wspace=0.0, hspace=0.0)

        k = 0
        for i in range(4):
            for j in range(2):
                axs = fig.add_subplot(gs[k])

                if j == 0:
                    axs.imshow(
                        all_profiles[i, :, :, wave_indice[l]],
                        cmap='gray',
                        origin='lower',
                        vmin=vmin[l][j],
                        vmax=vmax[l][j]
                    )
                    axs.scatter([ref_y], [ref_x], marker='+', color=color[i])
                    if i == 0:
                        axs.text(
                            0.1, 0.9,
                            r'Observed',
                            transform=axs.transAxes,
                            color='white',
                            fontsize='xx-small'
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
                        vmin=vmin[l][j],
                        vmax=vmax[l][j]
                    )
                    if i == 0:
                        axs.text(
                            0.1, 0.9,
                            r'Synthesized',
                            transform=axs.transAxes,
                            color='white',
                            fontsize='xx-small'
                        )

                axs.set_xticks([])
                axs.set_xticklabels([])
                axs.set_yticks([])
                axs.set_yticklabels([])

                k += 1

        if l == 0:
            fig.suptitle(r'$Ca\;II\;K_{2V}$', fontsize='small')
            fig.savefig('CaIIK2v_fit.pdf', format='pdf', dpi=300)
        elif l == 1:
            fig.suptitle(r'$Ca\;II\;K_{3}$', fontsize='small')
            fig.savefig('CaIIK3_fit.pdf', format='pdf', dpi=300)
        else:
            fig.suptitle(r'$Ca\;II\;K_{2R}$', fontsize='small')
            fig.savefig('CaIIK2r_fit.pdf', format='pdf', dpi=300)

    plt.cla()

    plt.clf()

    plt.close('all')

    fig, axs = plt.subplots(4, 3, figsize=(2.47 * 2, 2 * 2.47 * 4 / 3))

    for i in range(4):
        for j in range(3):
            if j == 0:
                axs[i][j].scatter(wave_3933[:-1] - 3933.682, all_profiles[i, ref_x, ref_y, 0:29], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_3933[:-1] - 3933.682, syn_profiles[i, ref_x, ref_y, 0:29], linestyle='--', color=color[i], linewidth=0.5)
                axs[i][j].set_ylabel(r'$I/I_{c}$')
                if i == 0:
                    axs[i][j].set_title(r'$Ca II K$')
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda(\AA)$')
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                axs[i][j].set_xticklabels([-0.5, 0, 0.5])
            elif j == 1:
                axs[i][j].scatter(wave_8542 - 8542.09, all_profiles[i, ref_x, ref_y, 30:30+20], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_8542 - 8542.09, syn_profiles[i, ref_x, ref_y, 30:30+20], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'$Ca\;II\;8542\;\AA$')
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda(\AA)$')
                axs[i][j].set_xticks([-1, 0, 1])
                axs[i][j].set_xticklabels([-1, 0, 1])
                axs[i][j].set_xlim(-2.4, 2.4)
            else:
                axs[i][j].scatter(wave_6173 - 6173.3340, all_profiles[i, ref_x, ref_y, 30 + 20:30 + 20 + 14], color=color[i], s=size/2, marker='+')
                axs[i][j].plot(wave_6173 - 6173.3340, syn_profiles[i, ref_x, ref_y, 30 + 20:30 + 20 + 14], linestyle='--', color=color[i], linewidth=0.5)
                if i == 0:
                    axs[i][j].set_title(r'$Fe\;I\;6173\;\AA$')
                if i == 3:
                    axs[i][j].set_xlabel(r'$\Delta\lambda(\AA)$')
                axs[i][j].set_xticks([-0.5, 0, 0.5])
                axs[i][j].set_xticklabels([-0.5, 0, 0.5])
                axs[i][j].set_xlim(-0.7, 0.7)

    fig.tight_layout()

    fig.savefig('Profile_fits.pdf', format='pdf', dpi=300)
