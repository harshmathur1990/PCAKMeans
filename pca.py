import sys
import h5py
import sunpy.io.fits
import numpy as np
from sklearn.decomposition import IncrementalPCA


def do_PCA(filename):

    data, header = sunpy.io.fits.read(filename, memmap=True)[0]

    sys.stdout.write(
        '{} opened for reading\n'.format(filename)
    )

    new_arr = np.lib.stride_tricks.as_strided(
        data,
        shape=(100 * 1236 * 1848, 30),
        strides=(4, 9136512)
    )

    pca = IncrementalPCA(n_components=30)

    sys.stdout.write(
        'About to perform PCA fit_transform\n'
    )

    principalComponents = pca.fit_transform(new_arr)

    sys.stdout.write(
        'Done with perform PCA fit_transform\n'
    )

    f = h5py.File('pca.h5', 'w')

    sys.stdout.write(
        'Opened pca.h5 for writing\n'
    )

    f['components_'] = pca.components_

    f['principalComponents'] = principalComponents

    f['explained_variance_'] = pca.explained_variance_

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_

    f['singular_values_'] = pca.singular_values_

    f['mean_'] = pca.mean_

    f['var_'] = pca.var_

    f['noise_variance_'] = pca.noise_variance_

    f['n_components_'] = pca.n_components_

    f['n_samples_seen_'] = pca.n_samples_seen_

    f.close()

    sys.stdout.write('Done\n')


if __name__ == '__main__':
    filename = sys.argv[1]
    do_PCA(filename)
