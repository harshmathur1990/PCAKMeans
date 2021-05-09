import sys
import numpy as np
import h5py
import sunpy.io
import matplotlib.pyplot as plt
from helita.io.lp import *

file_list = [
    '/data/harsh/sub_fov_result_kmeans_whole_data.h5'
]

selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'

whole_data = np.zeros((7 * 1236 * 1848, 30 + 20 + 14))

data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

whole_data[:, 0:30] = np.transpose(
    data[selected_frames][:, 0],
    axes=(0, 2, 3, 1)
).reshape(
    7 * 1236 * 1848,
    30
)

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

whole_data[:, 30:30 + 20] = np.transpose(
    data[selected_frames][:, 0],
    axes=(0, 2, 3, 1)
).reshape(
    7 * 1236 * 1848,
    20
)

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

whole_data[:, 30 + 20:30 + 20 + 14] = np.transpose(
    data[selected_frames][:, 0],
    axes=(0, 2, 3, 1)
).reshape(
    7 * 1236 * 1848,
    14
)

wave_3933 = np.array(
    [
        3932.78952, 3932.85488, 3932.92024, 3932.9856 , 3933.05096,
        3933.11632, 3933.18168, 3933.24704, 3933.3124 , 3933.37776,
        3933.44312, 3933.50848, 3933.57384, 3933.6392 , 3933.70456,
        3933.76992, 3933.83528, 3933.90064, 3933.966  , 3934.03136,
        3934.09672, 3934.16208, 3934.22744, 3934.2928 , 3934.35816,
        3934.42352, 3934.48888, 3934.55424, 3934.6196
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

cont_value = [2.4434714e-05, 4.2277254e-08, 4.054384e-08]
cont_array = np.zeros(30 + 20 + 14)
cont_array[0:30] = cont_value[0]
cont_array[30:30 + 20] = cont_value[1]
cont_array[30 + 20: 30 + 20 + 14] = cont_value[2]
in_bins_3950 = np.linspace(0, 0.6, 1000)
in_bins_8542 = np.linspace(0, 1, 1000)
in_bins_6173 = np.linspace(0, 1.3, 1000)


red = '#ec5858'
brown = '#fd8c04'
yellow = '#edf285'
blue = '#93abd3'


def get_farthest(a, center):
    global whole_data
    all_profiles = whole_data[a] / cont_array
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index]


def plot_profiles():

    for file in file_list:

        sys.stdout.write('Processing {}\n'.format(file))

        f = h5py.File(
            file,
            'r'
        )

        labels = f['columns']['final_labels'][selected_frames].reshape(
            7 * 1236 * 1848
        )

        plt.close('all')

        plt.clf()

        plt.cla()

        fig1, ax1 = plt.subplots(10, 10, figsize=(19.2, 10.8), sharey='col')
        fig2, ax2 = plt.subplots(10, 10, figsize=(19.2, 10.8), sharey='col')
        fig3, ax3 = plt.subplots(10, 10, figsize=(19.2, 10.8), sharey='col')

        k = 0

        for i in range(10):
            for j in range(10):

                a = np.where(labels == k)[0]

                center = f['columns']['rps'][k] / cont_array

                farthest_profile = get_farthest(a, center)

                H1, xedge1, yedge1 = np.histogram2d(
                    np.tile(wave_3933, a.shape[0]),
                    whole_data[a, 0:29].flatten() / cont_value[0],
                    bins=(wave_3933, in_bins_3950)
                )

                H2, xedge2, yedge2 = np.histogram2d(
                    np.tile(wave_8542, a.shape[0]),
                    whole_data[a, 30:30 + 20].flatten() / cont_value[1],
                    bins=(wave_8542, in_bins_8542)
                )

                H3, xedge3, yedge3 = np.histogram2d(
                    np.tile(wave_6173, a.shape[0]),
                    whole_data[a, 30 + 20: 30 + 20 + 14].flatten() / cont_value[2],
                    bins=(wave_6173, in_bins_6173)
                )

                ax1[i][j].plot(
                    wave_3933,
                    center[0:29],
                    color='black',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax1[i][j].plot(
                    wave_3933,
                    farthest_profile[0:29],
                    color='black',
                    linewidth=0.5,
                    linestyle='dotted'
                )

                ax2[i][j].plot(
                    wave_8542,
                    center[30:30 + 20],
                    color='black',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax2[i][j].plot(
                    wave_8542,
                    farthest_profile[30:30 + 20],
                    color='black',
                    linewidth=0.5,
                    linestyle='dotted'
                )
                ax3[i][j].plot(
                    wave_6173,
                    center[30 + 20:30 + 20 + 14],
                    color='black',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax3[i][j].plot(
                    wave_6173,
                    farthest_profile[30 + 20:30 + 20 + 14],
                    color='black',
                    linewidth=0.5,
                    linestyle='dotted'
                )

                X1, Y1 = np.meshgrid(xedge1, yedge1)

                X2, Y2 = np.meshgrid(xedge2, yedge2)

                X3, Y3 = np.meshgrid(xedge3, yedge3)

                ax1[i][j].pcolormesh(X1, Y1, H1.T, cmap='Greys')

                ax1[i][j].set_ylim(0, 0.6)

                ax1[i][j].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax1[i][j].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax1[i][j].text(
                    0.2,
                    0.6,
                    'n = {} %'.format(
                        np.round(a.size * 100 / 15988896, 4)
                    ),
                    transform=ax1[i][j].transAxes
                )

                ax1[i][j].text(
                    0.3,
                    0.8,
                    'RP {}'.format(k),
                    transform=ax1[i][j].transAxes
                )

                ax2[i][j].pcolormesh(X2, Y2, H2.T, cmap='Greys')

                ax2[i][j].set_ylim(0, 1)

                ax2[i][j].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax2[i][j].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax2[i][j].text(
                    0.2,
                    0.6,
                    'n = {} %'.format(
                        np.round(a.size * 100 / 15988896, 4)
                    ),
                    transform=ax2[i][j].transAxes
                )

                ax2[i][j].text(
                    0.3,
                    0.8,
                    'RP {}'.format(k),
                    transform=ax2[i][j].transAxes
                )

                ax3[i][j].pcolormesh(X3, Y3, H3.T, cmap='Greys')

                ax3[i][j].set_ylim(0, 1.3)

                ax3[i][j].tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax3[i][j].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False  # labels along the bottom edge are off
                )

                ax3[i][j].text(
                    0.2,
                    0.6,
                    'n = {} %'.format(
                        np.round(a.size * 100 / 15988896, 4)
                    ),
                    transform=ax3[i][j].transAxes
                )

                ax3[i][j].text(
                    0.3,
                    0.8,
                    'RP {}'.format(k),
                    transform=ax3[i][j].transAxes
                )

                k += 1

        fig1.savefig(
            '3950_RPs100.png',
            format='png',
            dpi=100
        )

        sys.stdout.write('Saved {}\n'.format('3950_RPs100.eps'))

        fig2.savefig(
            '8542_RPs100.png',
            format='png',
            dpi=100
        )

        sys.stdout.write('Saved {}\n'.format('8542_RPs100.eps'))

        fig3.savefig(
            '6173_RPs100.png',
            format='png',
            dpi=100
        )

        sys.stdout.write('Saved {}\n'.format('6173_RPs100.eps'))

        f.close()


if __name__ == '__main__':
    sys.stdout.write('Procedure Started\n')
    plot_profiles()
