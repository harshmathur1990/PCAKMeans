# import sys
from astropy.io import fits
import numpy as np
import h5py
import scipy.spatial
import matplotlib.pyplot as plt


file_list = [
    '/data/harsh/out_100_tol_1en6.h5',
    '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5',
    '/data/harsh/out_100_n_iter_10000_tol_1en5_sample_weights.h5',
    '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en6_sample_weights.h5'
]

primary_hdu = fits.open(
    '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits',
    memmap=False
)[0]
data, header = primary_hdu.data, primary_hdu.header
selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
data = data[selected_frames]
data[np.where(data < 0)] = 0
fo = h5py.File('/home/harsh/stic/shocks_rps/merged_rps_mean.nc', 'r')
cont_value = 2.4434714e-05
red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'


def plot_profiles():

    for file in file_list:

        f = h5py.File(file, 'r')

        labels = f['labels_'][()].reshape(7, 1236, 1848)

        error = np.zeros(100)

        for k in range(100):

            a, b, c = np.where(labels == k)

            center = f['rps'][k] / cont_value

            dist = scipy.spatial.distance.cdist(
                data[a, 0, :, b, c] / cont_value,
                center[np.newaxis, :]
            )

            difference = np.mean(dist)

            error[k] = difference

        np.savetxt(file + '_error.txt', error)

        plt.close('all')

        plt.clf()

        plt.cla()

        plt.hist(error, bins=30)

        mn = np.mean(error)

        sd = np.std(error)

        plt.text(
            0.4,
            0.7,
            'Std: {}'.format(sd),
            transform=plt.gca().transAxes
        )

        plt.text(
            0.4,
            0.9,
            'Mean: {}'.format(mn),
            transform=plt.gca().transAxes
        )

        plt.savefig(file + '_error.png', format='png', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()

        f.close()


if __name__ == '__main__':
    plot_profiles()
