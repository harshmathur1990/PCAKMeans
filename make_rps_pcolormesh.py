import sys
from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt

file_list = [
    '/data/harsh/out_100_tol_1en6.h5',
    '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5',
    '/data/harsh/out_100_n_iter_10000_tol_1en5_sample_weights.h5',
    '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en6_sample_weights.h5'
]
primary_hdu = fits.open(
    '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits',
    memmap=True
)[0]
data, header = primary_hdu.data, primary_hdu.header
selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
data = data[selected_frames]
data[np.where(data < 0)] = 0
fo = h5py.File('/home/harsh/stic/shocks_rps/merged_rps_mean.nc', 'r')
# labels = f['final_labels'][()]
wave = fo['wav'][4:33]
cont_value = 2.4434714e-05
in_bins = np.linspace(0, 0.6, 1000)
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


# def get_closest(a, b, c, center):
#     global data
#     all_profiles = data[a, 0, :-1, b, c] / cont_value
#     difference = np.sqrt(
#         np.sum(
#             np.square(
#                 np.subtract(
#                     all_profiles,
#                     center
#                 )
#             ),
#             axis=1
#         )
#     )
#     index = np.argsort(difference)[0]
#     return all_profiles[index]


def plot_profiles():

    for file in file_list:

        sys.stdout.write('Processing {}\n'.format(file))

        f = h5py.File(
            file,
            'r'
        )

        labels = f['labels_'][()].reshape(7, 1236, 1848)

        plt.close('all')

        plt.clf()

        plt.cla()

        fig, ax = plt.subplots(10, 10, figsize=(27, 18), sharey='col')

        k = 0

        for i in range(10):
            for j in range(10):

                a, b, c = np.where(labels == k)

                center = f['rps'][k, :-1] / cont_value

                farthest_profile = get_farthest(a, b, c, center)

                ax[i][j].plot(
                    wave,
                    center,
                    color='black',
                    linewidth=0.5,
                    linestyle='solid'
                )

                ax[i][j].plot(
                    wave,
                    farthest_profile,
                    color='black',
                    linewidth=0.5,
                    linestyle='dotted'
                )

                H, xedge, yedge = np.histogram2d(
                    np.tile(wave, a.shape[0]),
                    data[a, 0, :-1, b, c].flatten() / cont_value,
                    bins=(wave, in_bins)
                )

                X, Y = np.meshgrid(xedge, yedge)

                ax[i][j].pcolormesh(X, Y, H.T, cmap='Greys')

                ax[i][j].set_ylim(0, 0.6)

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

                ax[i][j].text(
                    0.2,
                    0.6,
                    'n = {} %'.format(
                        np.round(a.size * 100 / 15988896, 4)
                    ),
                    transform=ax[i][j].transAxes
                )

                ax[i][j].text(
                    0.3,
                    0.8,
                    'RP {}'.format(k),
                    transform=ax[i][j].transAxes
                )

                k += 1

        plt.savefig(
            file + '_RPs100.png',
            format='png',
            dpi=700
        )

        sys.stdout.write('Saved {}\n'.format(file + '_RPs100.png'))

        f.close()


if __name__ == '__main__':
    sys.stdout.write('Procedure Started\n')
    plot_profiles()
