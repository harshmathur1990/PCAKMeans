import sys
import numpy as np
import sunpy.io
import h5py
from helita.io.lp import *
import daal4py as d4p
import tables as tb
from mpi4py import MPI


kmeans_output_dir = '/data/harsh/kmeans_output'


# def prepare_data():
#     input_file_3950 = 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
#     input_file_8542 = 'nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
#     input_file_6173 = 'nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
#     old_kmeans_file = 'new_kmeans/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
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

#     first_concate = np.concatenate(
#         (
#             framerows_3950,
#             framerows_8542
#         ),
#         axis=1
#     )

#     del framerows_3950, framerows_8542

#     whole_data = np.concatenate(
#         (
#             first_concate,
#             framerows_6173
#         ),
#         axis=1
#     )

#     del framerows_6173

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

    accuracyThreshold = 1e-6

    f = h5py.File(filename, 'r')

    data = f['prepared_data_weighted']

    centroids = f['clusters'][()]

    rpp = int(data.shape[0] / d4p.num_procs())

    data_local = data[rpp * d4p.my_procid(): rpp * d4p.my_procid() + rpp, :]

    sys.stdout.write(
        'Process: {} Finished Loading data, proceeding for kmeans\n'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(
        nClusters,
        maxIter,
        accuracyThreshold=accuracyThreshold,
        distributed=True
    )

    result = algo.compute(data_local, centroids)

    algo = d4p.kmeans(nClusters, 0, assignFlag=True)

    assignments = algo.compute(data_local, result.centroids).assignments

    f.close()

    if d4p.my_procid() == 0:
        selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
        input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
        input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
        input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'

        data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

        whole_data = np.zeros((rpp, 30 + 20 + 14))

        whole_data[:, 0:30] = np.transpose(
            data[selected_frames][:, 0],
            axes=(0, 2, 3, 1)
        ).reshape(
            7 * 1236 * 1848,
            30
        )

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

        whole_data[:, 30:30 + 20] = np.transpose(
            data[selected_frames][:, 0],
            axes=(0, 2, 3, 1)
        ).reshape(
            7 * 1236 * 1848,
            20
        )

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

        whole_data[:, 30 + 20:30 + 20 + 14] = np.transpose(
            data[selected_frames][:, 0],
            axes=(0, 2, 3, 1)
        ).reshape(
            7 * 1236 * 1848,
            14
        )

        for i in range(1, d4p.num_procs()):
            comm.send(
                {
                    'whole_data': whole_data[rpp * i: rpp * i + rpp, :]
                },
                dest=i,
                tag=3
            )
        sys.stdout.write(
            'Process 0: Distributed actual data\n'
        )
        whole_data = whole_data[rpp * i: rpp * i + rpp, :]
    else:
        data_dict = comm.recv(
            source=0,
            tag=3,
            status=status
        )
        whole_data = data_dict['whole_data']
        sys.stdout.write(
            'Process {}: Received actual data\n'.format(
                d4p.my_procid()
            )
        )

    local_clusters = np.zeros((100, 30 + 20 + 14))
    total_numbers = np.zeros(100)
    for i in range(100):
        ind = np.where(assignments == i)[0]

        local_clusters[i] = np.sum(whole_data[ind], 0)
        total_numbers[i] = ind.size

    return (total_numbers, local_clusters, assignments, result)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    status = MPI.Status()

    filename = '/data/harsh/qhole_data_kmeans.nc'

    d4p.daalinit()

    (total_numbers, local_clusters, assignments, result) = do_kmeans(filename)

    if d4p.my_procid() == 0:

        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='w', title='KMeans Data')

        gcolumns = f.create_group('/', "columns", "KMeans arrays")

        f.create_array(gcolumns, 'assignments', np.zeros((assignments.shape[0] * d4p.num_procs())), "Labels")

        labels = f.root.columns.assignments

        labels[0:assignments.shape[0]] = assignments

        f.create_array(gcolumns, 'centroids', result.centroids, "Centroids")

        f.create_array(gcolumns, 'objectiveFunction', result.objectiveFunction, "Objective Function")

        f.create_array(gcolumns, 'nIterations', result.nIterations, "Number of Iterations")

        f.close()

        sys.stdout.write(
            'Process 0: Created File {}\n'.format(
                '/data/harsh/resut_kmeans_whole_data.h5'
            )
        )

        k = d4p.num_procs() - 2

        while (k >= 0):
            comm.recv(
                source=MPI.ANY_SOURCE,
                tag=0,
                status=status
            )
            sender = status.Get_source()
            data_dict = comm.recv(
                source=sender,
                tag=1,
                status=status
            )

            total_numbers = np.add(total_numbers, data_dict['total_numbers'])
            local_clusters = np.add(local_clusters, data_dict['local_clusters'])

            k -= 1

        rps = local_clusters / total_numbers

        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='a')

        f.create_array(f.root.columns, 'rps', rps, "Representative Profiles")

        f.close()

        sys.stdout.write(
            'Process {}: Updated File {}\n'.format(
                d4p.my_procid(),
                '/data/harsh/resut_kmeans_whole_data.h5'
            )
        )

    else:
        comm.send({}, dest=0, tag=0)
        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='a')
        labels = f.root.columns.assignments
        labels[assignments.shape[0] * d4p.my_procid(): assignments.shape[0] * d4p.my_procid() + assignments.shape[0]] = assignments
        f.close()
        sys.stdout.write(
            'Process {}: Updated File {}\n'.format(
                d4p.my_procid(),
                '/data/harsh/resut_kmeans_whole_data.h5'
            )
        )
        comm.send({'total_numbers':total_numbers, 'local_clusters': local_clusters}, dest=0, tag=1)
