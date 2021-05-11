import sys
import numpy as np
import h5py
import sunpy.io
import matplotlib.pyplot as plt
from helita.io.lp import *

file = '/data/harsh/sub_fov_result_kmeans_whole_data_same_weights.h5'

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
in_bins_3950 = np.linspace(0, 0.6, 1000)
in_bins_8542 = np.linspace(0, 1.3, 1000)
in_bins_6173 = np.linspace(0.4, 2, 1000)


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

    plt.close('all')

    plt.clf()

    plt.cla()

    fig01, ax01 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig02, ax02 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig03, ax03 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig04, ax04 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig05, ax05 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig06, ax06 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig07, ax07 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig08, ax08 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig09, ax09 = plt.subplots(10, 3, figsize=(8.27, 11.69))
    fig10, ax10 = plt.subplots(10, 3, figsize=(8.27, 11.69))

    fig = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10]
    ax = [ax01, ax02, ax03, ax04, ax05, ax06, ax07, ax08, ax09, ax10]

    k = 0
    axgtr = 0

    for i in range(10):

        axtr = 0

        for j in range(10):

            a = np.where(labels == k)[0]

            center = rps[k] / cont_array

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

            ax[axgtr][axtr][0].plot(
                wave_3933,
                center[0:29],
                color='black',
                linewidth=0.5,
                linestyle='solid'
            )

            ax[axgtr][axtr][0].plot(
                wave_3933,
                farthest_profile[0:29],
                color='black',
                linewidth=0.5,
                linestyle='dotted'
            )

            ax[axgtr][axtr][1].plot(
                wave_8542,
                center[30:30 + 20],
                color='black',
                linewidth=0.5,
                linestyle='solid'
            )

            ax[axgtr][axtr][1].plot(
                wave_8542,
                farthest_profile[30:30 + 20],
                color='black',
                linewidth=0.5,
                linestyle='dotted'
            )
            ax[axgtr][axtr][2].plot(
                wave_6173,
                center[30 + 20:30 + 20 + 14],
                color='black',
                linewidth=0.5,
                linestyle='solid'
            )

            ax[axgtr][axtr][2].plot(
                wave_6173,
                farthest_profile[30 + 20:30 + 20 + 14],
                color='black',
                linewidth=0.5,
                linestyle='dotted'
            )

            X1, Y1 = np.meshgrid(xedge1, yedge1)

            X2, Y2 = np.meshgrid(xedge2, yedge2)

            X3, Y3 = np.meshgrid(xedge3, yedge3)

            ax[axgtr][axtr][0].pcolormesh(X1, Y1, H1.T, cmap='Greys')

            ax[axgtr][axtr][0].set_ylim(0, 0.6)

            ax[axgtr][axtr][0].text(
                0.2,
                0.6,
                'n = {} %'.format(
                    np.round(a.size * 100 / labels.size, 4)
                ),
                transform=ax[axgtr][axtr][0].transAxes,
                fontsize=8
            )

            ax[axgtr][axtr][0].text(
                0.3,
                0.8,
                'RP {}'.format(k),
                transform=ax[axgtr][axtr][0].transAxes,
                fontsize=8
            )

            ax[axgtr][axtr][1].pcolormesh(X2, Y2, H2.T, cmap='Greys')

            ax[axgtr][axtr][1].set_ylim(0, 1.3)

            ax[axgtr][axtr][0].set_xticklabels([])
            ax[axgtr][axtr][1].set_xticklabels([])
            ax[axgtr][axtr][2].set_xticklabels([])

            # ax[axgtr][axtr][0].set_aspect(1.0 /  ax[axgtr][axtr][0].get_data_ratio(), adjustable='box')
            # ax[axgtr][axtr][1].set_aspect(1.0 /  ax[axgtr][axtr][1].get_data_ratio(), adjustable='box')
            # ax[axgtr][axtr][2].set_aspect(1.0 /  ax[axgtr][axtr][2].get_data_ratio(), adjustable='box')

            ax[axgtr][axtr][1].text(
                0.2,
                0.6,
                'n = {} %'.format(
                    np.round(a.size * 100 / labels.size, 4)
                ),
                transform=ax[axgtr][axtr][1].transAxes,
                fontsize=8
            )

            ax[axgtr][axtr][1].text(
                0.3,
                0.8,
                'RP {}'.format(k),
                transform=ax[axgtr][axtr][1].transAxes,
                fontsize=8
            )

            ax[axgtr][axtr][2].pcolormesh(X3, Y3, H3.T, cmap='Greys')

            ax[axgtr][axtr][2].set_ylim(0.4, 2)

            ax[axgtr][axtr][2].text(
                0.2,
                0.6,
                'n = {} %'.format(
                    np.round(a.size * 100 / labels.size, 4)
                ),
                transform=ax[axgtr][axtr][2].transAxes,
                fontsize=8
            )

            ax[axgtr][axtr][2].text(
                0.3,
                0.8,
                'RP {}'.format(k),
                transform=ax[axgtr][axtr][2].transAxes,
                fontsize=8
            )

            k += 1

            axtr += 1

        axgtr += 1

    for i in range(10):
        savename = 'SW_KMeans_100_{}_{}.png'.format(
            name, i
        )
        fig[i].tight_layout()
        fig[i].savefig(
            savename,
            format='png',
            dpi=300
        )
        sys.stdout.write(
            'Saved {}\n'.format(
                savename
            )
        )


def plot_profiles():
    global whole_data

    sys.stdout.write('Processing {}\n'.format(file))

    whole_data, n, o, p = get_data()
    f = h5py.File(file, 'r')
    labels = f['columns']['assignments'][()]
    rps = f['columns']['rps'][()]
    actual_plotting(labels, rps, name='output')
    f.close()

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
