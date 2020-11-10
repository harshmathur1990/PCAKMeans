import sys
from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt


f = h5py.File('/Volumes/Harsh 9599771751/Oslo Work/out_45.h5', 'r')
primary_hdu = fits.open(
    '/Volumes/Harsh 9599771751/colabd/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits',
    memmap=False
)[0]
data, header = primary_hdu.data, primary_hdu.header
data[np.where(data < 0)] = 0
fo = h5py.File('/Volumes/Harsh 9599771751/Oslo Work/merged_rps.nc', 'r')
labels = f['final_labels'][()]
wave = fo['wav'][4:33]
cont_value = 2.4434714e-05
in_bins = np.linspace(0, 0.3, 1000)
red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'


def plot_profiles():

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, ax = plt.subplots(5, 9, figsize=(18, 12), sharey='col')

    k = 0

    mean_profile = fo['profiles'][0, 0, 3, 4:33, 0]

    for i in range(5):
        for j in range(9):

            sys.stdout.write('i: {}, j: {}\n'.format(i, j))

            a, b, c = np.where(labels == k)

            center = np.mean(data[a, 0, :-1, b, c], axis=0) / cont_value

            ax[i][j].plot(wave, center, color=red)

            ax[i][j].plot(wave, mean_profile, color=brown)

            H, xedge, yedge = np.histogram2d(
                np.tile(wave, a.shape[0]),
                data[a, 0, :-1, b, c].flatten() / cont_value,
                bins=(wave, in_bins)
            )

            X, Y = np.meshgrid(xedge, yedge)

            ax[i][j].pcolormesh(X, Y, H.T, cmap='gray')

            ax[i][j].set_ylim(0, 0.3)

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

    plt.savefig('/data/harsh/RPs.png', format='png', dpi=100)


if __name__ == '__main__':
    plot_profiles()
