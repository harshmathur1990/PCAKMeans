import sys
import numpy as np
# import sunpy.io
import h5py
# from helita.io.lp import *
import daal4py as d4p


kmeans_output_dir = '/data/harsh/kmeans_output'


# def prepare_data():
#     input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
#     input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
#     input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
#     old_kmeans_file = '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
#     selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
#     y = 0.011995
#     x = 0.022555
#     weights = np.ones(30 + 20 + 14) * y
#     weights[10:20] = x
#     weights[30 + 4:30 + 16] = x

#     ## 3950
#     data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]
#     framerows_3950 = np.transpose(
#         data[selected_frames][:, 0],
#         axes=(0, 2, 3, 1)
#     ).reshape(
#         7 * 1236 * 1848,
#         30
#     )

#     mn_3950 = np.mean(framerows_3950, axis=0)
#     sd_3950 = np.std(framerows_3950, axis=0)

#     ## 8542
#     sh, dt, header = getheader(input_file_8542)
#     data = np.memmap(
#         input_file_8542,
#         mode='r',
#         shape=sh,
#         dtype=dt,
#         order='F',
#         offset=512
#     )

#     data = np.transpose(
#         data.reshape(1848, 1236, 100, 4, 20),
#         axes=(2, 3, 4, 1, 0)
#     )

#     framerows_8542 = np.transpose(
#         data[selected_frames][:, 0],
#         axes=(0, 2, 3, 1)
#     ).reshape(
#         7 * 1236 * 1848,
#         20
#     )

#     mn_8542 = np.mean(framerows_8542, axis=0)
#     sd_8542 = np.std(framerows_8542, axis=0)

#     ## 6173
#     sh, dt, header = getheader(input_file_6173)
#     data = np.memmap(
#         input_file_6173,
#         mode='r',
#         shape=sh,
#         dtype=dt,
#         order='F',
#         offset=512
#     )

#     data = np.transpose(
#         data.reshape(1848, 1236, 100, 4, 14),
#         axes=(2, 3, 4, 1, 0)
#     )

#     framerows_6173 = np.transpose(
#         data[selected_frames][:, 0],
#         axes=(0, 2, 3, 1)
#     ).reshape(
#         7 * 1236 * 1848,
#         14
#     )

#     mn_6173 = np.mean(framerows_6173, axis=0)
#     sd_6173 = np.std(framerows_6173, axis=0)

#     framerows_3950 = (framerows_3950 - mn_3950) / sd_3950
#     framerows_8542 = (framerows_8542 - mn_8542) / sd_8542
#     framerows_6173 = (framerows_6173 - mn_6173) / sd_6173

#     whole_data = np.zeros((framerows_3950.shape[0], 30 + 20 + 14), dtype=np.float64)

#     whole_data[:, 0:30] = framerows_3950
#     whole_data[:, 30:30 + 20] = framerows_8542
#     whole_data[:, 30 + 20:30 + 20 + 14] = framerows_6173

#     del framerows_3950, framerows_8542, framerows_6173

#     whole_data *= weights

#     f = h5py.File(old_kmeans_file, 'r')

#     labels = f['labels_'][()]

#     clusters = np.zeros((100, 30 + 20 + 14))

#     for i in range(100):
#         ind = np.where(labels == i)[0]
#         clusters[i] = np.mean(whole_data[ind], 0)

#     f = h5py.File('qhole_data_kmeans.nc', 'w')

#     f['prepared_data_weighted'] = whole_data

#     f['weights'] = weights

#     f['clusters'] = clusters

#     f.close()


def do_kmeans(filename, method='plusPlusDense'):
    nClusters = 100

    maxIter = 10000

    f = h5py.File(filename, 'r')

    data = f['prepared_data_weighted']

    centroids = f['clusters'][()]

    rpp = int(data.shape[0] / d4p.num_procs())

    data = data[rpp * d4p.my_procid(): rpp * d4p.my_procid() + rpp, :]

    f.close()

    sys.stdout.write(
        'Process: {} Finished Loading data, proceeding for kmeans\n'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(nClusters, maxIter, distributed=True)

    result = algo.compute(data, centroids)

    algo = d4p.kmeans(nClusters, 0, assignFlag=True)

    assignments = algo.compute(data, result.centroids).assignments

    return (assignments, result)


if __name__ == '__main__':

    filename = '/data/harsh/qhole_data_kmeans.nc'

    d4p.daalinit()

    (assignments, result) = do_kmeans(filename)

    if d4p.my_procid() == 0:
        f = h5py.File('/data/harsh/resut_kmeans_whole_data.h5', 'w')

        f['assignments'] = assignments

        f['centroids'] = result.centroids

        f['objectiveFunction'] = result.objectiveFunction

        f.close()

        sys.stdout.write(
            'Saved File {}\n'.format(
                '/data/harsh/resut_kmeans_whole_data.h5'
            )
        )
