import sys
import numpy as np
import h5py
import sunpy.io
import matplotlib.pyplot as plt
from helita.io.lp import *
from mpl_toolkits.axisartist.axislines import Subplot
import matplotlib.gridspec as gridspec

file = '/data/harsh/sub_fov_result_kmeans_whole_data_inertia_inverted_weights.h5'

selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
old_kmeans_file = '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
mask_file_crisp = '/data/harsh/crisp_chromis_mask_2019-06-06.fits'
input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'

mask, _  = sunpy.io.fits.read(mask_file_crisp, memmap=True)[0]

mask = np.transpose(mask, axes=(2, 1, 0))


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

total_shock_profiles = np.array(
    list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles)
)

red = '#ec5858'
brown = '#fd8c04'
yellow = '#edf285'
blue = '#93abd3'
whole_data = None

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


def get_max_min(a):
    global whole_data
    all_profiles = whole_data[a] / cont_array
    return all_profiles[:, 0:29].max(), all_profiles[:, 30:30 + 20].max(), all_profiles[:, 30 + 20:30 + 20 + 14].max(), all_profiles[:, 0:29].min(), all_profiles[:, 30:30 + 20].min(), all_profiles[:, 30 + 20:30 + 20 + 14].min()


def get_data():
    n, o, p = np.where(mask[selected_frames] == 1)

    whole_data = np.zeros((n.size, 30 + 20 + 14))

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[:, 0:30] = data[selected_frames][n, 0, :, o, p]

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

    whole_data[:, 30:30 + 20] = data[selected_frames][n, 0, :, o, p]

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

    whole_data[:, 30 + 20:30 + 20 + 14] = data[selected_frames][n, 0, :, o, p]

    return whole_data, n, o, p


def actual_plotting(labels, rps, name='guess'):

    global whole_data

    k = 0

    for m in range(2):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(4.135, 5.845))

        gs = gridspec.GridSpec(10, 5)

        gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

        for i in range(10):

            for j in range(5):

                if k in total_shock_profiles:
                    color = '#001E6C'
                    cm = 'Blues'
                else:
                    color = 'black'
                    cm = 'Greys'

                sys.stdout.write('{}\n'.format(k))

                ax1 = fig.add_subplot(gs[k if k < 50 else k-50], label='1')

                a = np.where(labels == k)[0]

                center = rps[k] / cont_array

                farthest_profile = get_farthest(a, center)

                b, c, d, e, f, g = get_max_min(a)

                max_3950, min_3950  = b, e
                max_8542, min_8542  = c, f
                max_6173, min_6173  = d, g

                min_3950 = min_3950 * 0.9
                max_3950 = max_3950 * 1.1

                in_bins_3950 = np.linspace(min_3950, max_3950, 1000)

                H1, xedge1, yedge1 = np.histogram2d(
                    np.tile(wave_3933, a.shape[0]),
                    whole_data[a, 0:29].flatten() / cont_value[0],
                    bins=(wave_3933, in_bins_3950)
                )

                ax1.plot(
                    wave_3933,
                    center[0:29],
                    color=color,
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax1.plot(
                    wave_3933,
                    farthest_profile[0:29],
                    color=color,
                    linewidth=0.5,
                    linestyle='dotted'
                )


                X1, Y1 = np.meshgrid(xedge1, yedge1)

                ax1.pcolormesh(X1, Y1, H1.T, cmap=cm)

                ax1.set_ylim(min_3950, max_3950)

                ax1.text(
                    0.2,
                    0.6,
                    'n = {} %'.format(
                        np.round(a.size * 100 / labels.size, 4)
                    ),
                    transform=ax1.transAxes,
                    fontsize=8
                )

                ax1.text(
                    0.3,
                    0.8,
                    'RP {}'.format(k),
                    transform=ax1.transAxes,
                    fontsize=8
                )

                ax1.set_xticks([3933.682])
                ax1.set_xticklabels([])

                y_ticks = [
                    np.round(
                        (min_3950 + max_3950) / 3,
                        2
                    ),
                    np.round(
                        2 * (min_3950 + max_3950) / 3,
                        2
                    )
                ]

                ax1.set_yticks(y_ticks)
                ax1.set_yticklabels(y_ticks)

                ax1.tick_params(axis="y",direction="in", pad=-30)

                ax2 = fig.add_subplot(
                    gs[k if k < 50 else k-50], label='2', frame_on=False
                )

                ax2.plot(
                    wave_8542,
                    center[30:30 + 20],
                    color='#420516',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax2.set_xticks([])
                ax2.set_yticks([])

                ax3 = fig.add_subplot(
                    gs[k if k < 50 else k-50], label='3', frame_on=False
                )

                ax3.plot(
                    wave_6173,
                    center[30 + 20:30 + 20 + 14],
                    color='#FC5404',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax3.set_xticks([])
                ax3.set_yticks([])

                k += 1

        fig.savefig(
            'RPs_{}.pdf'.format(k),
            format='pdf',
            dpi=300
        )


def plot_profiles():
    global whole_data

    whole_data, n, o, p = get_data()
    # f = h5py.File(file, 'r')
    # labels = f['columns']['assignments'][()]
    # rps = f['columns']['rps'][()]
    # actual_plotting(labels, rps, name='output')
    # f.close()

    f = h5py.File(old_kmeans_file, 'r')
    labels = f['new_final_labels'][selected_frames][n, o, p]
    rps = np.zeros((100, 64))
    for i in range(100):
        ind = np.where(labels == i)[0]
        rps[i] = np.mean(whole_data[ind], 0)

    actual_plotting(labels, rps, name='guess')
    f.close()


if __name__ == '__main__':
    sys.stdout.write('Procedure Started\n')
    plot_profiles()
