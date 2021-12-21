import sys
from pathlib import Path
import numpy as np
import h5py
import sunpy.io
from helita.io.lp import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# mask_file_crisp = Path('/home/harsh/OsloAnalysis/crisp_chromis_mask_2019-06-06.fits')
input_file_3950 = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
input_file_8542 = Path('/home/harsh/OsloAnalysis/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')
input_file_6173 = Path('/home/harsh/OsloAnalysis/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')
input_file_6173_blos = Path('/home/harsh/OsloAnalysis/Blos.6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')
input_file_hmi_blos = Path('/home/harsh/OsloAnalysis/hmimag_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.icube')

# mask, _  = sunpy.io.fits.read(mask_file_crisp, memmap=True)[0]

# mask = np.transpose(mask, axes=(2, 1, 0))

selected_frames = np.array([7])

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


@np.vectorize
def get_relative_velocity_Ca_8542(wavelength):
    return wavelength - 8542.09


def get_data(hmi=False):

    whole_data = np.zeros((selected_frames.shape[0], 64, 1236, 1848))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[0, 0:30] = data[7, 0]

    sh, dt, header = getheader(input_file_6173)
    data = np.memmap(
        input_file_6173,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 14),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[0, 30:30 + 14] = data[7, 0]

    sh, dt, header = getheader(input_file_8542)
    data = np.memmap(
        input_file_8542,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 20),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[0, 30 + 14:30 + 14 + 20] = data[7, 0]

    sh, dt, header = getheader(input_file_6173_blos)

    data = np.memmap(
        input_file_6173_blos,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    b6173 = np.transpose(
        data,
        axes=(2, 1, 0)
    )

    hmi_mag = None

    if hmi:
        sh, dt, header = getheader(input_file_hmi_blos)

        data = np.memmap(
            input_file_hmi_blos,
            mode='r',
            shape=sh,
            dtype=dt,
            order='F',
            offset=512
        )

        hmi_mag = np.transpose(
            data,
            axes=(2, 1, 0)
        )
    return whole_data, b6173, hmi_mag


def plot_one_image():
    whole_data, b6173, hmi_mag = get_data()
    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(1, 1)

    gs.update(left=0.1, right=1, top=1, bottom=0.1, wspace=0.0, hspace=0.0)

    axs = fig.add_subplot(gs[0])

    extent = [596.31, 666.312, -35.041, 11.765]

    fov_1_mask = np.zeros((1236, 1848))

    fov_1_mask[662:712, 708:758] = 1
    axs.text(
        638 / 1848,
        662 / 1236,
        'A',
        transform=axs.transAxes,
        color='#4E79A7'
    )

    ca_k_indice = np.array([14, 12])

    axs.imshow(whole_data[0, ca_k_indice[1], :, :], cmap='gray', origin='lower', extent=extent)

    axs.text(
        0.05, 0.91,
        r'$\mathrm{{Ca\;II\;K\;}}{}\mathrm{{\;m\AA}}$'.format(
            np.round(
                get_relative_velocity(
                    wave_3933[ca_k_indice[1]]
                ) * 1000,
                1
            )
        ),
        transform=axs.transAxes,
        color='white'
    )

    axs.contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')

    axs.yaxis.set_minor_locator(MultipleLocator(1))

    axs.xaxis.set_minor_locator(MultipleLocator(1))

    axs.tick_params(direction='in', which='both', color='white')

    axs.set_ylabel('y [arcsec]')

    axs.set_xlabel('x [arcsec]')

    write_path = Path(
        '/home/harsh/Shocks Paper'
    )
    fig.savefig(write_path / 'FOV_one_image.pdf', format='pdf', dpi=300)
    fig.savefig(write_path / 'FOV_one_image.png', format='png', dpi=300)

    plt.close('all')
    plt.clf()
    plt.cla()


def plot_fov_images():
    whole_data, b6173, hmi_mag = get_data()

    plt.close('all')
    plt.clf()
    plt.cla()

    fig = plt.figure(figsize=(7, 6.77))

    gs = gridspec.GridSpec(3, 2)

    gs.update(left=0.1, right=1, top=1, bottom=0.07, wspace=0.0, hspace=0.0)

    axs = list()

    k = 0

    for i in range(3):
        axsi = list()
        for j in range(2):
            axsi.append(fig.add_subplot(gs[k]))

            k += 1
        axs.append(axsi)


    extent = [596.31, 666.312, -35.041, 11.765]

    fov_1_mask = np.zeros((1236, 1848))
    fov_2_mask = np.zeros((1236, 1848))
    fov_3_mask = np.zeros((1236, 1848))
    fov_4_mask = np.zeros((1236, 1848))
    fov_5_mask = np.zeros((1236, 1848))
    fov_6_mask = np.zeros((1236, 1848))
    fov_7_mask = np.zeros((1236, 1848))
    fov_8_mask = np.zeros((1236, 1848))
    fov_9_mask = np.zeros((1236, 1848))
    fov_10_mask = np.zeros((1236, 1848))

    #fov A
    fov_1_mask[662:712, 708:758] = 1
    axs[1][0].text(
        638 / 1848,
        662 / 1236,
        'A',
        transform=axs[1][0].transAxes,
        color='#4E79A7'
    )
    # #fov B
    # fov_2_mask[915:965, 1072:1122] = 1
    # axs[1][0].text(
    #     1002 / 1848,
    #     915 / 1236,
    #     'B',
    #     transform=axs[1][0].transAxes,
    #     color='#F28E2B'
    # )

    #fov C
    fov_3_mask[486:536, 974:1024] = 1
    axs[1][0].text(
        1034 / 1848,
        486 / 1236,
        'B',
        transform=axs[1][0].transAxes,
        color='#E15759'
    )

    #fov D
    fov_4_mask[582:632, 627:677] = 1
    axs[1][0].text(
        557 / 1848,
        582 / 1236,
        'C',
        transform=axs[1][0].transAxes,
        color='#76B7B2'
    )

    #fov E
    fov_5_mask[810:860, 335:385] = 1
    axs[1][0].text(
        265 / 1848,
        810 / 1236,
        'D',
        transform=axs[1][0].transAxes,
        color='#59A14F'
    )

    #fov F
    fov_6_mask[455:505, 940:990] = 1
    axs[1][0].text(
        870 / 1848,
        455 / 1236,
        'E',
        transform=axs[1][0].transAxes,
        color='#EDC948'
    )

    # #fov G
    # fov_7_mask[95:145, 600:650] = 1
    # axs[1][0].text(
    #     530 / 1848,
    #     95 / 1236,
    #     'G',
    #     transform=axs[1][0].transAxes,
    #     color='#B07AA1'
    # )

    #fov H
    fov_8_mask[315:365, 855:905] = 1
    axs[1][0].text(
        785 / 1848,
        315 / 1236,
        'F',
        transform=axs[1][0].transAxes,
        color='#FF9DA7'
    )

    #fov I
    fov_9_mask[600:650, 1280:1330] = 1
    axs[1][0].text(
        1220 / 1848,
        600 / 1236,
        'G',
        transform=axs[1][0].transAxes,
        color='#9C755F'
    )

    #fov J
    fov_10_mask[535:585, 715:765] = 1
    axs[1][0].text(
        785 / 1848,
        535 / 1236,
        'H',
        transform=axs[1][0].transAxes,
        color='#BAB0AC'
    )

    ca_k_indice = np.array([14, 12])
    ca_8_indice = np.array([10, 9])
    axs[0][0].imshow(whole_data[0, 29, :, :], cmap='gray', origin='lower', extent=extent)
    im = axs[0][1].imshow(b6173[7], cmap='gray', origin='lower', extent=extent, vmin=-300, vmax=300)

    axs[1][0].imshow(whole_data[0, ca_k_indice[0], :, :], cmap='gray', origin='lower', extent=extent)
    axs[1][1].imshow(whole_data[0, ca_k_indice[1], :, :], cmap='gray', origin='lower', extent=extent)

    axs[2][0].imshow(whole_data[0, 30 + 14 + ca_8_indice[0], :, :], cmap='gray', origin='lower', extent=extent)
    axs[2][1].imshow(whole_data[0, 30 + 14 + ca_8_indice[1], :, :], cmap='gray', origin='lower', extent=extent)

    axs[0][0].text(0.05, 0.91, r'(a) Continuum 4000 $\mathrm{\AA}$', transform=axs[0][0].transAxes, color='white')
    axs[0][1].text(
        0.05, 0.91,
        r'(b) $B_{{\mathrm{LOS}}}$',
        transform=axs[0][1].transAxes,
        color='white'
    )
    axs[1][0].text(
        0.05, 0.91,
        r'(c) $\mathrm{{Ca\;II\;K\;}}+{}\mathrm{{\;m\AA}}$'.format(
            np.round(
                get_relative_velocity(
                    wave_3933[ca_k_indice[0]]
                ) * 1000,
                1
            )
        ),
        transform=axs[1][0].transAxes,
        color='white'
    )
    axs[1][1].text(
        0.05, 0.91,
        r'(d) $\mathrm{{Ca\;II\;K\;}}{}\mathrm{{\;m\AA}}$'.format(
            np.round(
                get_relative_velocity(
                    wave_3933[ca_k_indice[1]]
                ) * 1000,
                1
            )
        ),
        transform=axs[1][1].transAxes,
        color='white'
    )
    axs[2][0].text(
        0.05, 0.91,
        r'(e) $\mathrm{{Ca\;II\;8542\;\AA\;}}+{}\mathrm{{\;m\AA}}$'.format(
            np.round(
                get_relative_velocity_Ca_8542(
                    wave_8542[ca_8_indice[0]]
                ) * 1000,
                1
            )
        ),
        transform=axs[2][0].transAxes,
        color='white'
    )
    axs[2][1].text(
        0.05, 0.91,
        r'(f) $\mathrm{{Ca\;II\;8542\;\AA\;}}{}\mathrm{{\;m\AA}}$'.format(
            np.round(
                get_relative_velocity_Ca_8542(
                    wave_8542[ca_8_indice[1]]
                ) * 1000,
                1
            )
        ),
        transform=axs[2][1].transAxes,
        color='white'
    )

    cbaxes = inset_axes(
        axs[0][1],
        width="50%",
        height="8%",
        loc=1,
        borderpad=1
    )
    cbar = fig.colorbar(
        im,
        cax=cbaxes,
        ticks=[-300, 300],
        orientation='horizontal'
    )

    # cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='white')

    axs[0][0].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')
    axs[0][1].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')
    axs[1][0].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')
    axs[1][1].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')
    axs[2][0].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')
    axs[2][1].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='#4E79A7')

    # axs[0][0].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')
    # axs[0][1].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')
    # axs[1][0].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')
    # axs[1][1].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')
    # axs[2][0].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')
    # axs[2][1].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='#F28E2B')

    axs[0][0].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')
    axs[0][1].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')
    axs[1][0].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')
    axs[1][1].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')
    axs[2][0].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')
    axs[2][1].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='#E15759')

    axs[0][0].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')
    axs[0][1].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')
    axs[1][0].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')
    axs[1][1].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')
    axs[2][0].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')
    axs[2][1].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='#76B7B2')

    axs[0][0].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')
    axs[0][1].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')
    axs[1][0].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')
    axs[1][1].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')
    axs[2][0].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')
    axs[2][1].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='#59A14F')

    axs[0][0].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')
    axs[0][1].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')
    axs[1][0].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')
    axs[1][1].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')
    axs[2][0].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')
    axs[2][1].contour(fov_6_mask, levels=0, extent=extent, origin='lower', colors='#EDC948')

    # axs[0][0].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')
    # axs[0][1].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')
    # axs[1][0].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')
    # axs[1][1].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')
    # axs[2][0].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')
    # axs[2][1].contour(fov_7_mask, levels=0, extent=extent, origin='lower', colors='#B07AA1')

    axs[0][0].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')
    axs[0][1].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')
    axs[1][0].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')
    axs[1][1].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')
    axs[2][0].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')
    axs[2][1].contour(fov_8_mask, levels=0, extent=extent, origin='lower', colors='#FF9DA7')

    axs[0][0].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')
    axs[0][1].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')
    axs[1][0].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')
    axs[1][1].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')
    axs[2][0].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')
    axs[2][1].contour(fov_9_mask, levels=0, extent=extent, origin='lower', colors='#9C755F')

    axs[0][0].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')
    axs[0][1].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')
    axs[1][0].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')
    axs[1][1].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')
    axs[2][0].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')
    axs[2][1].contour(fov_10_mask, levels=0, extent=extent, origin='lower', colors='#BAB0AC')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].yaxis.set_minor_locator(MultipleLocator(1))
    #
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[2][1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].tick_params(direction='in', which='both', color='white')
    axs[0][1].tick_params(direction='in', which='both', color='white')
    axs[1][0].tick_params(direction='in', which='both', color='white')
    axs[1][1].tick_params(direction='in', which='both', color='white')
    axs[2][0].tick_params(direction='in', which='both', color='white')
    axs[2][1].tick_params(direction='in', which='both', color='white')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[2][1].set_yticklabels([])

    axs[0][0].set_ylabel('y [arcsec]')
    axs[1][0].set_ylabel('y [arcsec]')
    axs[2][0].set_ylabel('y [arcsec]')

    axs[2][0].set_xlabel('x [arcsec]')
    axs[2][1].set_xlabel('x [arcsec]')

    write_path = Path(
        '/home/harsh/Shocks Paper'
    )
    fig.savefig(write_path / 'FOV.pdf', format='pdf', dpi=300)
    fig.savefig(write_path / 'FOV.png', format='png', dpi=300)

    plt.close('all')
    plt.clf()
    plt.cla()


if __name__ == '__main__':
    # plot_fov_images()
    plot_one_image()
