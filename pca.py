import sys
import h5py
import sunpy.io
import numpy as np
from sklearn.decomposition import PCA


def do_PCA(filename):

    data, header = sunpy.io.read_file(filename)[0]

    sys.stdout.write(
        '{} opened for reading\n'.format(filename)
    )

    new_arr = np.lib.stride_tricks.as_strided(
        data,
        shape=(100 * 1236 * 1848, 30),
        strides=(4, 9136512)
    )

    pca = PCA(n_components=30)

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

    f['means_'] = pca.means_

    f['components_'] = pca.components_

    f['principalComponents'] = principalComponents

    f['explained_variance_'] = pca.explained_variance_

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_

    f['singular_values_'] = pca.singular_values_

    f['noise_variance_'] = pca.noise_variance_

    f['n_components_'] = pca.n_components_

    f['n_features_'] = pca.n_features_

    f['n_samples_'] = pca.n_samples_

    f.close()

    sys.stdout.write('Done\n')


if __name__ == '__main__':
    filename = sys.argv[1]
    do_PCA(filename)
