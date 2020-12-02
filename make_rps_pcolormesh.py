import sys
from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt


f = h5py.File('/data/harsh/out_100_0.5_0.5.h5', 'r')
primary_hdu = fits.open(
    '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits',
    memmap=False
)[0]
data, header = primary_hdu.data, primary_hdu.header
data[np.where(data < 0)] = 0
fo = h5py.File('/home/harsh/stic/shocks_rps/merged_rps_mean.nc', 'r')
labels = f['final_labels'][()]
wave = fo['wav'][4:33]
cont_value = 2.4434714e-05
in_bins = np.linspace(0, 0.4, 1000)
red = '#ec5858'
brown = '#fd8c04'
yellow = '#edf285'
blue = '#93abd3'


def get_farthest(a, b, c, center):
    global data
    all_profiles = data[a, 0, :-1, b, c] / cont_value
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


def get_closest(a, b, c, center):
    global data
    all_profiles = data[a, 0, :-1, b, c] / cont_value
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
    index = np.argsort(difference)[0]
    return all_profiles[index]


def plot_profiles():

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, ax = plt.subplots(10, 10, figsize=(27, 18), sharey='col')

    k = 0

    # mean_profile = f['rps'][4, :-1] / cont_value

    for i in range(10):
        for j in range(10):

            sys.stdout.write('i: {}, j: {}\n'.format(i, j))

            a, b, c = np.where(labels == k)

            center = f['rps'][k, :-1] / cont_value

            medprof = np.median(data[a, 0, :-1, b, c] / cont_value, axis=0)

            farthest_profile = get_farthest(a, b, c, center)

            closest_profile = get_closest(a, b, c, center)

            ax[i][j].plot(wave, center, color=red, linewidth=0.5)

            ax[i][j].plot(wave, medprof, color=brown, linewidth=0.5)

            ax[i][j].plot(wave, farthest_profile, color=yellow, linewidth=0.5)

            ax[i][j].plot(wave, closest_profile, color=blue, linewidth=0.5)

            H, xedge, yedge = np.histogram2d(
                np.tile(wave, a.shape[0]),
                data[a, 0, :-1, b, c].flatten() / cont_value,
                bins=(wave, in_bins)
            )

            X, Y = np.meshgrid(xedge, yedge)

            ax[i][j].pcolormesh(X, Y, H.T, cmap='Greys')

            ax[i][j].set_ylim(0, 0.4)

            ax[i][j].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False  # labels along the bottom edge are off
            )

            ax[i][j].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False  # labels along the bottom edge are off
            )

            k += 1

    plt.savefig('/data/harsh/RPs100_0.5_0.5.png', format='png', dpi=700)


if __name__ == '__main__':
    plot_profiles()