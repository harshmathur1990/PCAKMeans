import sys
from pathlib import Path
import numpy as np
import h5py
import sunpy.io
from helita.io.lp import *
import matplotlib.pyplot as plt


# mask_file_crisp = Path('/home/harsh/OsloAnalysis/crisp_chromis_mask_2019-06-06.fits')
input_file_3950 = Path('/home/harsh/OsloAnalysis/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits')
input_file_8542 = Path('/home/harsh/OsloAnalysis/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')
input_file_6173 = Path('/home/harsh/OsloAnalysis/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube')


# mask, _  = sunpy.io.fits.read(mask_file_crisp, memmap=True)[0]

# mask = np.transpose(mask, axes=(2, 1, 0))

selected_frames = np.array([0])

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


def get_data():

    whole_data = np.zeros((selected_frames.shape[0], 64, 1236, 1848))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[0, 0:30] = data[0, 0]

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

    whole_data[0, 30:30 + 14] = data[0, 0]

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

    whole_data[0, 30 + 14:30 + 14 + 20] = data[0, 0]

    return whole_data


def plot_fov_images():
    whole_data = get_data()

    # X, Y = np.meshgrid(np.arange(1848), np.arange(1236))

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(2, 2, figsize=(8.27, 5.845))

    extent = [596.31, 666.312, -35.041, 11.765]

    fov_1_mask = np.zeros((1236, 1848))
    fov_2_mask = np.zeros((1236, 1848))
    fov_3_mask = np.zeros((1236, 1848))
    fov_4_mask = np.zeros((1236, 1848))
    fov_5_mask = np.zeros((1236, 1848))

    #fov 1
    fov_1_mask[662:712, 708:758] = 1

    #fov 2
    fov_2_mask[520:570, 715:765] = 1

    #fov 3
    fov_3_mask[640:690, 1000:1050] = 1

    #fov 4
    fov_4_mask[915:965, 1072:1122] = 1

    #fov 5
    fov_5_mask[486:536, 974:1024] = 1

    axs[0][0].imshow(whole_data[0, 29, :, :], cmap='gray', origin='lower', extent=extent)
    axs[0][1].imshow(whole_data[0, 13, :, :], cmap='gray', origin='lower', extent=extent)

    axs[1][0].imshow(whole_data[0, 15, :, :], cmap='gray', origin='lower', extent=extent)
    axs[1][1].imshow(whole_data[0, 30 + 14 + 9, :, :], cmap='gray', origin='lower', extent=extent)

    axs[0][0].text(0.0, 0.1, r'(a) Continuum 4000 $\AA$', transform=axs[0][0].transAxes, color='white')
    axs[0][1].text(0.0, 0.1, r'(b) Ca II k $-3.26\;km/sec$', transform=axs[0][1].transAxes, color='white')
    axs[1][0].text(0.0, 0.1, r'(c) Ca II k $+6.7\;km/sec$', transform=axs[1][0].transAxes, color='white')
    axs[1][1].text(0.0, 0.1, r'(d) Ca II 8542 core', transform=axs[1][1].transAxes, color='white')

    axs[0][0].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='blue')
    axs[0][1].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='blue')
    axs[1][0].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='blue')
    axs[1][1].contour(fov_1_mask, levels=0, extent=extent, origin='lower', colors='blue')

    axs[0][0].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='red')
    axs[0][1].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='red')
    axs[1][0].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='red')
    axs[1][1].contour(fov_2_mask, levels=0, extent=extent, origin='lower', colors='red')

    axs[0][0].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='yellow')
    axs[0][1].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='yellow')
    axs[1][0].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='yellow')
    axs[1][1].contour(fov_3_mask, levels=0, extent=extent, origin='lower', colors='yellow')

    axs[0][0].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='green')
    axs[0][1].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='green')
    axs[1][0].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='green')
    axs[1][1].contour(fov_4_mask, levels=0, extent=extent, origin='lower', colors='green')

    axs[0][0].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='pink')
    axs[0][1].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='pink')
    axs[1][0].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='pink')
    axs[1][1].contour(fov_5_mask, levels=0, extent=extent, origin='lower', colors='pink')

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[1][1].set_yticklabels([])

    axs[0][0].set_ylabel('y[arcsec]')
    axs[1][0].set_ylabel('y[arcsec]')

    axs[1][0].set_xlabel('x[arcsec]')
    axs[1][1].set_xlabel('x[arcsec]')

    fig.tight_layout()
    fig.savefig('FOV.eps', format='eps', dpi=300)
    fig.savefig('FOV.png', format='png', dpi=300)

    plt.close('all')
    plt.clf()
    plt.cla()