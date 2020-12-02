import sys
import sunpy.io
import numpy as np
import h5py
from sklearn.cluster import KMeans


data, header = sunpy.io.fits.read(
    '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits',
    memmap=True
)[0]

selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])


def do_kmeans():

    sys.stdout.write('Started...\n')

    framerows = np.transpose(
        data[selected_frames][:, 0],
        axes=(0, 2, 3, 1)
    ).reshape(
        7 * 1236 * 1848,
        30
    )

    sys.stdout.write('Selected Data...\n')

    mn = np.mean(framerows, axis=0)

    sd = np.std(framerows, axis=0)

    normframerows = (framerows - mn) / sd

    sys.stdout.write('Normalised Data...\n')

    weights = np.ones(30) * 0.025

    weights[10:20] = 0.05

    normframerows *= weights

    sys.stdout.write('Multiplied Weights, will do KMeans now...\n')

    model = KMeans(
        n_clusters=100,
        max_iter=10000,
        tol=1e-6,
        n_jobs=32
    )

    model._n_threads = 32

    model.fit(normframerows)

    sys.stdout.write('fitted KMeans, saving now...\n')

    f = h5py.File(
        '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5', 'w'
    )

    f['cluster_centers_'] = model.cluster_centers_

    f['labels_'] = model.labels_

    f['inertia_'] = model.inertia_

    f['n_iter_'] = model.n_iter_

    f.close()

    sys.stdout.write('Saved Everything...\n')


if __name__ == '__main__':
    do_kmeans()
